#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import torch
from loguru import logger

import tensorrt as trt
from damo.apis.detector_inference_trt import inference
from damo.config.base import parse_config
from damo.dataset import build_dataloader, build_dataset
from damo.utils import setup_logger, synchronize


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo trt engine eval')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument('-t',
                        '--trt',
                        default=None,
                        type=str,
                        help='trt for eval')
    parser.add_argument('--conf', default=None, type=float, help='test conf')
    parser.add_argument('--nms',
                        default=None,
                        type=float,
                        help='test nms threshold')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='inference image batch nums')
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    parser.add_argument(
        '--end2end',
        action='store_true',
        help='trt inference with nms',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def trt_inference(config,
                  trt_name,
                  img_size,
                  batch_size=None,
                  conf=None,
                  nms=None,
                  end2end=False):

    # dist init
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    os.environ['WORLD_SIZE'] = '1'
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=0)
    synchronize()

    file_name = os.path.join(config.miscs.output_dir, config.miscs.exp_name)
    os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name,
                 distributed_rank=0,
                 mode='a')

    if conf is not None:
        config.model.head.nms_conf_thre = conf
    if nms is not None:
        config.model.head.nms_iou_thre = nms
    if batch_size is not None:
        config.test.batch_size = batch_size

    # set logs
    loggert = trt.Logger(trt.Logger.INFO)

    trt.init_libnvinfer_plugins(loggert, '')

    # initialize
    t = open(trt_name, 'rb')
    runtime = trt.Runtime(loggert)
    model = runtime.deserialize_cuda_engine(t.read())
    context = model.create_execution_context()

    # start evaluate
    output_folders = [None] * len(config.dataset.val_ann)

    if config.miscs.output_dir:
        for idx, dataset_name in enumerate(config.dataset.val_ann):
            output_folder = os.path.join(config.miscs.output_dir, 'inference',
                                         dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    val_dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
    val_loader = build_dataloader(val_dataset,
                                  config.test.augment,
                                  batch_size=config.test.batch_size,
                                  num_workers=config.miscs.num_workers,
                                  is_train=False,
                                  size_div=img_size)

    for output_folder, dataset_name, data_loader_val in zip(
            output_folders, config.dataset.val_ann, val_loader):
        inference(
            config,
            context,
            data_loader_val,
            dataset_name,
            iou_types=('bbox', ),
            box_only=False,
            output_folder=output_folder,
            end2end=end2end,
        )


@logger.catch
def main():
    args = make_parser().parse_args()
    config = parse_config(args.config_file)
    config.merge(args.opts)

    trt_inference(config,
                  args.trt,
                  args.img_size,
                  batch_size=args.batch_size,
                  conf=args.conf,
                  nms=args.nms,
                  end2end=args.end2end)


if __name__ == '__main__':
    main()
