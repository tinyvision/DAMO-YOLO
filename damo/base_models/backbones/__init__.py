# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from damo.base_models.backbones.tinynas_csp import load_tinynas_net as load_tinynas_net_csp
from damo.base_models.backbones.tinynas_res import load_tinynas_net as load_tinynas_net_res
from damo.base_models.backbones.tinynas_mob import load_tinynas_net as load_tinynas_net_mob


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'TinyNAS_res':
        return load_tinynas_net_res(backbone_cfg)
    elif name == 'TinyNAS_csp':
        return load_tinynas_net_csp(backbone_cfg)
    elif name == 'TinyNAS_mob':
        return load_tinynas_net_mob(backbone_cfg)
    else:
        print(f'{name} is not supported yet!')
