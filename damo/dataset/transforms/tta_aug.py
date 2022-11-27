# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch

from damo.dataset.transforms import transforms as T
from damo.structures.bounding_box import BoxList
from damo.structures.image_list import to_image_list
from damo.utils.boxes import filter_results


def im_detect_bbox_aug(model, images, device, config):
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(len(images)):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            if len(boxlists_ts[i]) == 0:
                # The first one is identity transform,
                # no need to resize the boxlist
                boxlists_ts[i].append(boxlist_t)
            else:
                # Resize the boxlist as the first one
                boxlists_ts[i].append(boxlist_t.resize(boxlists_ts[i][0].size))

    # Compute detections for the original image (identity transform)
    boxlists_i = im_detect_bbox(model, images, config.testing.input_min_size,
                                config.testing.input_max_size, device, config)
    add_preds_t(boxlists_i)

    # Perform detection on the horizontally flipped image
    if config.testing.augmentation.hflip:
        boxlists_hf = im_detect_bbox_hflip(model, images,
                                           config.testing.input_min_size,
                                           config.testing.input_max_size,
                                           device, config)
        add_preds_t(boxlists_hf)

    # Compute detections at different scales
    for scale in config.testing.augmentation.scales:
        max_size = config.testing.augmentation.scales_max_size
        boxlists_scl = im_detect_bbox_scale(model, images, scale, max_size,
                                            device, config)
        add_preds_t(boxlists_scl)

        if config.testing.augmentation.scales_hflip:
            boxlists_scl_hf = im_detect_bbox_scale(model,
                                                   images,
                                                   scale,
                                                   max_size,
                                                   device,
                                                   config,
                                                   hflip=True)
            add_preds_t(boxlists_scl_hf)

    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for i, boxlist_ts in enumerate(boxlists_ts):
        bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
        scores = torch.cat(
            [boxlist_t.get_field('scores') for boxlist_t in boxlist_ts])
        labels = torch.cat(
            [boxlist_t.get_field('labels') for boxlist_t in boxlist_ts])
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field('scores', scores)
        boxlist.add_field('labels', labels)
        boxlists.append(boxlist)

    # Apply NMS and limit the final detections
    results = []
    for boxlist in boxlists:
        results.append(
            filter_results(boxlist, config.model.head.num_classes,
                           config.testing.augmentation.nms_thres))

    return results


def im_detect_bbox(model, images, target_scale, target_max_size, device,
                   config):
    """
    Performs bbox detection on the original image.
    """
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.ToTensor(),
        T.Normalize(mean=config.dataset.input_pixel_mean,
                    std=config.dataset.input_pixel_std,
                    to_bgr255=config.dataset.input_to_bgr255)
    ])
    images = [transform(image)[0] for image in images]
    images = to_image_list(images, config.dataset.size_divisibility)
    return model(images.to(device))


def im_detect_bbox_hflip(model, images, target_scale, target_max_size, device,
                         config):
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.RandomHorizontalFlip(1.0),
        T.ToTensor(),
        T.Normalize(mean=config.dataset.input_pixel_mean,
                    std=config.dataset.input_pixel_std,
                    to_bgr255=config.dataset.input_to_bgr255)
    ])
    images = [transform(image)[0] for image in images]
    images = to_image_list(images, config.dataset.size_divisibility)
    boxlists = model(images.to(device))

    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists]
    return boxlists_inv


def im_detect_bbox_scale(model,
                         images,
                         target_scale,
                         target_max_size,
                         device,
                         config,
                         hflip=False):
    """
    Computes bbox detections at the given scale.
    Returns predictions in the scaled image space.
    """
    if hflip:
        boxlists_scl = im_detect_bbox_hflip(model, images, target_scale,
                                            target_max_size, device, config)
    else:
        boxlists_scl = im_detect_bbox(model, images, target_scale,
                                      target_max_size, device, config)
    return boxlists_scl
