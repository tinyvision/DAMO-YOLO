# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import cv2
import numpy as np


def debug_input_vis(imgs, targets, ids, train_loader):

    std = np.array([1.0, 1.0, 1.0]).reshape(3, 1, 1)
    mean = np.array([0.0, 0.0, 0.0]).reshape(3, 1, 1)

    n, c, h, w = imgs.shape
    for i in range(n):
        img = imgs[i, :, :, :].cpu()
        bboxs = targets[i].bbox.cpu().numpy()
        cls = targets[i].get_field('labels').cpu().numpy()
        if True:
            # if self.config.training_mosaic:
            img_id = train_loader.dataset._dataset.id_to_img_map[ids[i]]
        else:
            img_id = train_loader.dataset.id_to_img_map[ids[i]]

        img = np.clip(
            (img.numpy() * std + mean).transpose(1, 2,
                                                 0).copy().astype(np.uint8), 0,
            255)
        for bbox, obj_cls in zip(bboxs, cls):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(0, 0, 255),
                          thickness=2)
            cv2.putText(img, f'{obj_cls}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255))

        cv2.imwrite(f'visimgs/vis_{img_id}.jpg', img)
