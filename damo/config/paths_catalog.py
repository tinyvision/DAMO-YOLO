# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
"""Centralized catalog of paths."""
import os


class DatasetCatalog(object):
    DATA_DIR = 'datasets'
    DATASETS = {
        'objects365_train': {
            'img_dir': 'objects365/train',
            'ann_file': 'objects365/objects365_train.json'
        },
        'objects365_val': {
            'img_dir': 'objects365/val',
            'ann_file': 'objects365/objects365_val.json'
        },
        'coco_2017_train_mini': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017_minieven.json'
        },
        'coco_2017_train': {
            'img_dir': 'coco/train2017',
            'ann_file': 'coco/annotations/instances_train2017.json'
        },
        'coco_2017_val': {
            'img_dir': 'coco/val2017',
            'ann_file': 'coco/annotations/instances_val2017.json'
        },
        'coco_2017_test_dev': {
            'img_dir': 'coco/test2017',
            'ann_file': 'coco/annotations/image_info_test-dev2017.json'
        },
        'coco_2014_train': {
            'img_dir': 'coco/train2014',
            'ann_file': 'coco/annotations/instances_train2014.json'
        },
        'coco_2014_val': {
            'img_dir': 'coco/val2014',
            'ann_file': 'coco/annotations/instances_val2014.json'
        },
        'coco_2014_minival': {
            'img_dir': 'coco/val2014',
            'ann_file': 'coco/annotations/instances_minival2014.json'
        },
        'coco_2014_valminusminival': {
            'img_dir': 'coco/val2014',
            'ann_file': 'coco/annotations/instances_valminusminival2014.json'
        },
        'voc_2007_train': {
            'data_dir': 'voc/VOC2007',
            'split': 'train'
        },
        'voc_2007_train_cocostyle': {
            'img_dir': 'voc/VOC2007/JPEGImages',
            'ann_file': 'voc/VOC2007/Annotations/pascal_train2007.json'
        },
        'voc_2007_val': {
            'data_dir': 'voc/VOC2007',
            'split': 'val'
        },
        'voc_2007_val_cocostyle': {
            'img_dir': 'voc/VOC2007/JPEGImages',
            'ann_file': 'voc/VOC2007/Annotations/pascal_val2007.json'
        },
        'voc_2007_test': {
            'data_dir': 'voc/VOC2007',
            'split': 'test'
        },
        'voc_2007_test_cocostyle': {
            'img_dir': 'voc/VOC2007/JPEGImages',
            'ann_file': 'voc/VOC2007/Annotations/pascal_test2007.json'
        },
        'voc_2012_train': {
            'data_dir': 'voc/VOC2012',
            'split': 'train'
        },
        'voc_2012_train_cocostyle': {
            'img_dir': 'voc/VOC2012/JPEGImages',
            'ann_file': 'voc/VOC2012/Annotations/pascal_train2012.json'
        },
        'voc_2012_val': {
            'data_dir': 'voc/VOC2012',
            'split': 'val'
        },
        'voc_2012_val_cocostyle': {
            'img_dir': 'voc/VOC2012/JPEGImages',
            'ann_file': 'voc/VOC2012/Annotations/pascal_val2012.json'
        },
        'voc_2012_test': {
            'data_dir': 'voc/VOC2012',
            'split': 'test'
        }
    }

    @staticmethod
    def get(name):
        if 'coco' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='COCODataset',
                args=args,
            )
        elif 'voc' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs['data_dir']),
                split=attrs['split'],
            )
            return dict(
                factory='PascalVOCDataset',
                args=args,
            )
        elif 'objects365' in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs['img_dir']),
                ann_file=os.path.join(data_dir, attrs['ann_file']),
            )
            return dict(
                factory='Objects365',
                args=args,
            )
        return None
