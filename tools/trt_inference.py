# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

import tensorrt as trt
from cuda import cuda
from damo.config.base import parse_config
from damo.structures.bounding_box import BoxList
from damo.utils import postprocess, vis
from damo.utils.demo_utils import transform_img

COCO_CLASSES = []
for i in range(80):
    COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('trt engine inference')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument(
        '-t',
        '--trt_path',
        type=str,
        default='lightvision_small.trt',
        help='Input your trt model.',
    )
    parser.add_argument(
        '--end2end',
        action='store_true',
        help='trt inference with nms',
    )
    parser.add_argument(
        '-p',
        '--path',
        type=str,
        default='./assets/dog.jpg',
        help='Path to your input image.',
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='confidence threshould to filter the result.',
    )
    parser.add_argument(
        '--nms',
        type=float,
        default=0.6,
        help='nms threshould to filter the result.',
    )
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    # settings
    target_dtype = np.float32

    origin_img = np.asarray(Image.open(args.path).convert('RGB'))

    config = parse_config(args.config_file)

    # build transform
    img = transform_img(origin_img, args.img_size,
                        **config.test.augment.transform)
    img_np = np.asarray(img.tensors)

    if args.conf is not None:
        config.model.head.nms_conf_thre = args.conf
    if args.nms is not None:
        config.model.head.nms_iou_thre = args.nms

    # set logs
    loggert = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(loggert, '')

    # initialize
    t = open(args.trt_path, 'rb')
    runtime = trt.Runtime(loggert)

    model = runtime.deserialize_cuda_engine(t.read())
    context = model.create_execution_context()
    allocations = []
    inputs = []
    outputs = []
    for i in range(context.engine.num_bindings):
        is_input = False
        if context.engine.binding_is_input(i):
            is_input = True
        name = context.engine.get_binding_name(i)
        dtype = context.engine.get_binding_dtype(i)
        shape = context.engine.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        size = np.dtype(trt.nptype(dtype)).itemsize
        for s in shape:
            size *= s
        allocation = cuda.cuMemAlloc(size)
        binding = {
            'index': i,
            'name': name,
            'dtype': np.dtype(trt.nptype(dtype)),
            'shape': list(shape),
            'allocation': allocation,
            'size': size
        }
        allocations.append(allocation[1])
        if context.engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    input_batch = img_np.astype(target_dtype)

    trt_out = []
    for output in outputs:
        trt_out.append(np.zeros(output['shape'], output['dtype']))

    def predict(batch):  # result gets copied into output
        # transfer input data to device
        cuda.cuMemcpyHtoD(inputs[0]['allocation'][1],
                          np.ascontiguousarray(batch), int(inputs[0]['size']))
        # execute model
        context.execute_v2(allocations)
        # transfer predictions back
        for o in range(len(trt_out)):
            cuda.cuMemcpyDtoH(trt_out[o], outputs[o]['allocation'][1],
                              outputs[o]['size'])
        return trt_out

    pred_out = predict(input_batch)
    # trt with nms
    if args.end2end:
        nums = pred_out[0]
        boxes = pred_out[1]
        scores = pred_out[2]
        pred_classes = pred_out[3]
        batch_size = boxes.shape[0]
        output = [None for _ in range(batch_size)]
        for i in range(batch_size):
            img_h, img_w = img.image_sizes[i]
            boxlist = BoxList(torch.Tensor(boxes[i][:nums[i][0]]),
                              (img_w, img_h),
                              mode='xyxy')
            boxlist.add_field(
                'objectness',
                torch.Tensor(np.ones_like(scores[i][:nums[i][0]])))
            boxlist.add_field('scores', torch.Tensor(scores[i][:nums[i][0]]))
            boxlist.add_field('labels',
                              torch.Tensor(pred_classes[i][:nums[i][0]] + 1))
            output[i] = boxlist
    else:
        cls_scores = torch.Tensor(pred_out[0])
        bbox_preds = torch.Tensor(pred_out[1])
        output = postprocess(cls_scores, bbox_preds,
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
                  conf=config.model.head.nms_conf_thre,
                  class_names=COCO_CLASSES)

    output_folder = os.path.join(config.miscs.output_dir, 'trt_out')
    mkdir(output_folder)
    output_path = os.path.join(output_folder, os.path.split(args.path)[-1])
    logger.info('saved trt inference result into {}'.format(output_path))
    cv2.imwrite(output_path, out_img[:, :, ::-1])
