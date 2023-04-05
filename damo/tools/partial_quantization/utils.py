import os
import torch
import torch.nn as nn
import copy

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from damo.dataset import build_dataloader, build_dataset


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_module(model, submodule_key):
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


def module_quant_disable(ptq_model, k):
    verified_module = get_module(ptq_model, k)
    if hasattr(verified_module, '_input_quantizer'):
        verified_module._input_quantizer.disable()
    if hasattr(verified_module, '_weight_quantizer'):
        verified_module._weight_quantizer.disable()


def collect_stats(model, data_loader, batch_number, device='cuda'):
    """
      code mainly from https://github.com/NVIDIA/TensorRT/blob/99a11a5fcdd1f184739bb20a8c4a473262c8ecc8/tools/pytorch-quantization/examples/torchvision/classification_flow.py
      Feed data to the network and collect statistic
    """

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, data_tuple in enumerate(data_loader):
        images, targets, image_ids = data_tuple
        images = images.to(device)
        output = model(images)
        if i + 1 >= batch_number:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    """
      code mainly from https://github.com/NVIDIA/TensorRT/blob/99a11a5fcdd1f184739bb20a8c4a473262c8ecc8/tools/pytorch-quantization/examples/torchvision/classification_flow.py
      Load calib result
    """
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(F"{name:40}: {module}")
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)


def quantable_op_check(k, ops_to_quant):
    if ops_to_quant is None:
        return True

    if k in ops_to_quant:
        return True
    else:
        return False


def quant_model_init(ori_model, device):

    ptq_model = copy.deepcopy(ori_model)
    ptq_model.eval()
    ptq_model.to(device)
    quant_conv_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    quant_conv_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')

    quant_convtrans_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    quant_convtrans_desc_input = QuantDescriptor(num_bits=8, calib_method='histogram')

    for k, m in ptq_model.named_modules():
        if 'proj_conv' in k:
            print("Layer {} won't be quantized".format(k))
            continue

        if isinstance(m, nn.Conv2d):
            quant_conv = quant_nn.QuantConv2d(m.in_channels,
                                              m.out_channels,
                                              m.kernel_size,
                                              m.stride,
                                              m.padding,
                                              quant_desc_input = quant_conv_desc_input,
                                              quant_desc_weight = quant_conv_desc_weight)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(ptq_model, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            quant_convtrans = quant_nn.QuantConvTranspose2d(m.in_channels,
                                                       m.out_channels,
                                                       m.kernel_size,
                                                       m.stride,
                                                       m.padding,
                                                       quant_desc_input = quant_convtrans_desc_input,
                                                       quant_desc_weight = quant_convtrans_desc_weight)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(ptq_model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(m.kernel_size,
                                                      m.stride,
                                                      m.padding,
                                                      m.dilation,
                                                      m.ceil_mode,
                                                      quant_desc_input = quant_conv_desc_input)
            set_module(ptq_model, k, quant_maxpool2d)
        else:
            continue

    return ptq_model.to(device)


def post_train_quant(ori_model, calib_data_loader, calib_img_number, device):
    ptq_model = quant_model_init(ori_model, device)
    with torch.no_grad():
        collect_stats(ptq_model, calib_data_loader, calib_img_number, device)
        compute_amax(ptq_model, method='entropy')
    return ptq_model


def load_quanted_model(model, calib_weights_path, device):
    ptq_model = quant_model_init(model, device)
    ptq_model.load_state_dict(torch.load(calib_weights_path)['model'].state_dict())
    return ptq_model


def execute_partial_quant(ptq_model, ops_to_quant=None):
    for k, m in ptq_model.named_modules():
        if quantable_op_check(k, ops_to_quant):
            continue
        # enable full-precision
        if isinstance(m, quant_nn.QuantConv2d) or \
            isinstance(m, quant_nn.QuantConvTranspose2d) or \
            isinstance(m, quant_nn.QuantMaxPool2d):
            module_quant_disable(ptq_model, k)


def init_calib_data_loader(config):
    # init dataloader
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    os.environ['WORLD_SIZE'] = '1'
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=0)

    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset,
                                  config.test.augment,
                                  batch_size=config.test.batch_size,
                                  num_workers=config.miscs.num_workers,
                                  is_train=False,
                                  size_div=32)

    return val_loader[0]



