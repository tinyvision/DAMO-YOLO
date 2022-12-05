#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
import onnxruntime
from loguru import logger
from PIL import Image


from damo.config.base import parse_config
from damo.utils import postprocess, vis
from damo.utils.demo_utils import transform_img


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


COCO_CLASSES = []
for i in range(80):
    COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def make_parser():
    parser = argparse.ArgumentParser('damo eval')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument('-p',
                        '--path',
                        default='./assets/dog.jpg',
                        type=str,
                        help='path to image')
    parser.add_argument('--onnx',
                        default='./onnx/damoyolo_L25_S.onnx',
                        type=str,
                        help='onnx engine path')
    parser.add_argument('--conf',
                        default=0.6,
                        type=float,
                        help='conf of visualization')

    return parser


def build_onnx_engine(onnx_path):

    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    out_names = []
    out_shapes = []
    for idx in range(len(session.get_outputs())):
        out_names.append(session.get_outputs()[idx].name)
        out_shapes.append(session.get_outputs()[idx].shape)
    return session, input_name, input_shape[2:], out_names, out_shapes




@logger.catch
def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    args = make_parser().parse_args()

    origin_img = np.asarray(Image.open(args.path).convert('RGB'))
    config = parse_config(args.config_file)
    sess, input_name, input_size, _, _ = build_onnx_engine(args.onnx)

    config.dataset.size_divisibility = input_size[0]
    img = transform_img(origin_img, input_size[0],
                        **config.test.augment.transform)
    img_np = np.asarray(img.tensors)

    output_folder = './demo'
    mkdir(output_folder)

    output = sess.run(None, {input_name: img_np})

    scores = torch.Tensor(output[0])
    bboxes = torch.Tensor(output[1])
    output = postprocess(scores, bboxes,
        config.model.head.num_classes,
        config.model.head.nms_conf_thre,
        config.model.head.nms_iou_thre, img)

    ratio = min(origin_img.shape[0] / img.image_sizes[0][0],
                origin_img.shape[1] / img.image_sizes[0][1])
    bboxes = output[0].bbox * ratio
    scores = output[0].get_field('scores')
    cls_inds = output[0].get_field('labels')

    out_img = vis(origin_img,
                  bboxes,
                  scores,
                  cls_inds,
                  conf=args.conf,
                  class_names=COCO_CLASSES)

    output_path = os.path.join(output_folder, args.path.split('/')[-1])
    logger.info('saved onnx inference result into {}'.format(output_path))
    cv2.imwrite(output_path, out_img[:, :, ::-1])


if __name__ == '__main__':
    main()
