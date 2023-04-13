# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..core.ops import Focus, RepConv, SPPBottleneck, get_activation, DepthwiseConv
from damo.utils import make_divisible


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvKXBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, depthwise=False):
        super(ConvKXBN, self).__init__()
        if depthwise:
            self.conv1 = DepthwiseConv(in_channels=in_c,
                                       out_channels=out_c,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=(kernel_size-1) // 2,
                                       norm_cfg="bn",
                                       act="relu",
                                       order=("depthwise","pointwise")
                                       )
        else:
            self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size,
                               stride, (kernel_size - 1) // 2,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)

def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class MobileV3Block(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx',
                 depthwise=False,
                 use_se=False,
                 block_pos=None):
        super(MobileV3Block, self).__init__()
        self.stride = stride
        self.exp_ratio = 2.5
        if block_pos is not None:
            self.exp_ratio = 3.5 + (block_pos-1) * 0.5

        branch_features = math.ceil(out_c * self.exp_ratio)
        branch_features = make_divisible(branch_features)

        # assert (self.stride != 1) or (in_c == branch_features << 1)

        if use_se:
            SELayer = SEModule
        else:
            SELayer = nn.Identity

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            get_activation(act),
            depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=5,
                stride=self.stride,
                padding=2,
            ),
            nn.BatchNorm2d(branch_features),
            SELayer(branch_features),
            get_activation(act),
            nn.Conv2d(
                branch_features,
                out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
        )
        self.use_shotcut = self.stride == 1 and in_c == out_c

    def forward(self, x):
        if self.use_shotcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SuperResStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu',
                 reparam=False,
                 block_type='k1kx',
                 depthwise=False,
                 use_se=False,
                 block_pos=None,):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        basic_block = MobileV3Block
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = basic_block(in_channels,
                                     out_channels,
                                     btn_c,
                                     this_kernel_size,
                                     this_stride,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type,
                                     depthwise=depthwise,
                                     use_se=use_se,
                                     block_pos=block_pos,)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class TinyNAS(nn.Module):
    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 4, 5],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False,
                 depthwise=False,
                 use_se=False,):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()

        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus:
                    the_block = Focus(block_info['in'],
                                      block_info['out'],
                                      block_info['k'],
                                      act=act)
                else:
                    the_block = ConvKXBNRELU(3,
                                             block_info['out'],
                                             block_info['k'],
                                             2,
                                             act=act)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvK1KX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='k1kx',
                                         depthwise=depthwise,
                                         use_se=use_se,
                                         block_pos=idx)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='kxkx',
                                         depthwise=depthwise,
                                         use_se=use_se)
                self.block_list.append(the_block)
            else:
                raise NotImplementedError

    def init_weights(self, pretrain=None):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


def load_tinynas_net(backbone_cfg):
    # load masternet model to path
    import ast

    struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNAS(structure_info=struct_info,
                    out_indices=backbone_cfg.out_indices,
                    with_spp=backbone_cfg.with_spp,
                    use_focus=backbone_cfg.use_focus,
                    act=backbone_cfg.act,
                    reparam=backbone_cfg.reparam,
                    depthwise=backbone_cfg.depthwise,
                    use_se=backbone_cfg.use_se,)

    return model
