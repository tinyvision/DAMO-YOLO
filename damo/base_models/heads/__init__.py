# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from damo.base_models.heads import ZeroHead


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    else:
        raise NotImplementedError
