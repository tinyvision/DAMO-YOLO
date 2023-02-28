# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from damo.augmentations.scale_aware_aug import SA_Aug

from . import transforms as T


def build_transforms(start_epoch,
                     total_epochs,
                     no_aug_epochs,
                     iters_per_epoch,
                     num_workers,
                     batch_size,
                     num_gpus,
                     image_max_range=(640, 640),
                     flip_prob=0.5,
                     image_mean=[0, 0, 0],
                     image_std=[1., 1., 1.],
                     autoaug_dict=None,
                     keep_ratio=True):

    transform = [
        T.Resize(image_max_range, keep_ratio=keep_ratio),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ]

    if autoaug_dict is not None:
        transform += [
            SA_Aug(iters_per_epoch, start_epoch, total_epochs, no_aug_epochs,
                   batch_size, num_gpus, num_workers, autoaug_dict)
        ]

    transform = T.Compose(transform)

    return transform
