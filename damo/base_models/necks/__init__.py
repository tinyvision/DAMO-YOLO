# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .giraffe_fpn_btn import GiraffeNeckV2


def build_neck(cfg):
    neck_cfg = copy.deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'GiraffeNeckV2':
        return GiraffeNeckV2(**neck_cfg)
    else:
        raise NotImplementedError
