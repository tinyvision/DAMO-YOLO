# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
SADA = {
    'box_prob':
    0.3,
    'num_subpolicies':
    5,
    'scale_splits': [2048, 10240, 51200],
    'autoaug_params':
    (6, 9, 5, 3, 3, 4, 2, 4, 4, 4, 5, 2, 4, 1, 4, 2, 6, 4, 2, 2, 2, 6, 2, 2, 2,
     0, 5, 1, 3, 0, 8, 5, 2, 8, 7, 5, 1, 3, 3, 3),
}
Mosaic_Mixup = {
    'mosaic_prob': 1.0,
    'mosaic_scale': (0.1, 2.0),
    'mosaic_size': (640, 640),
    'mixup_prob': 1.0,
    'mixup_scale': (0.5, 1.5),
    'degrees': 10.0,
    'translate': 0.2,
    'shear': 2.0,
    'keep_ratio': False,
}

train_transform = {
    'image_mean': [0.0, 0.0, 0.0],
    'image_std': [1.0, 1.0, 1.0],
    'image_max_range': (640, 640),
    'flip_prob': 0.5,
    'keep_ratio': False,
    'autoaug_dict': SADA,
}
test_transform = {
    'image_mean': [0.0, 0.0, 0.0],
    'image_std': [1.0, 1.0, 1.0],
    'image_max_range': (640, 640),
    'flip_prob': 0.0,
    'keep_ratio': False,
}

train_aug = {
    'mosaic_mixup': Mosaic_Mixup,
    'transform': train_transform,
}

test_aug = {
    'transform': test_transform,
}
