#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 10
        self.miscs.ckpt_interval_epochs = 10
        # optimizer
        self.train.batch_size = 256
        self.train.base_lr_per_img = 0.001 / 64
        self.train.min_lr_ratio = 0.05
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5

        self.train.optimizer = {
            'name': "AdamW",
            'weight_decay': 1e-2,
            'lr': 4e-3,
            }

        # augment
        self.train.augment.transform.image_max_range = (416, 416)
        self.train.augment.transform.keep_ratio = False
        self.test.augment.transform.keep_ratio = False
        self.test.augment.transform.image_max_range = (416, 416)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)
        self.train.augment.mosaic_mixup.keep_ratio = False

        self.dataset.train_ann = ('coco_2017_train', )
        self.dataset.val_ann = ('coco_2017_val', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L20_k1kx_nano.txt')
        TinyNAS = {
            'name': 'TinyNAS_mob',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': False,
            'depthwise': True,
            'use_se': False,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 0.5,
            'hidden_ratio': 0.5,
            'in_channels': [80, 112, 160],
            'out_channels': [64, 128, 256],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
            'depthwise': True,
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 80,
            'in_channels': [64, 128, 256],
            'stacked_convs': 0,
            'reg_max': 7,
            'act': 'silu',
            'nms_conf_thre': 0.03,
            'nms_iou_thre': 0.65,
            'legacy': False,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
