#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import torch
from loguru import logger

from damo.base_models.core.ops import RepConv
from damo.apis.detector_inference import inference
from damo.config.base import parse_config
from damo.dataset import build_dataloader, build_dataset
from damo.detectors.detector import build_ddp_model, build_local_model
from damo.utils import fuse_model, get_model_info, setup_logger, synchronize


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo eval')

    # distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt for eval')
    parser.add_argument('--conf', default=None, type=float, help='test conf')
    parser.add_argument('--nms',
                        default=None,
                        type=float,
                        help='test nms threshold')
    parser.add_argument('--tsize',
                        default=None,
                        type=int,
                        help='test img size')
    parser.add_argument('--seed', default=None, type=int, help='eval seed')
    parser.add_argument(
        '--fuse',
        dest='fuse',
        default=False,
        action='store_true',
        help='Fuse conv and bn for testing.',
    )
    parser.add_argument(
        '--test',
        dest='test',
        default=False,
        action='store_true',
        help='Evaluating on test-dev set.',
    )  # TODO
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()

    device = 'cuda'
    config = parse_config(args.config_file)
    config.merge(args.opts)

    save_dir = os.path.join(config.miscs.output_dir, config.miscs.exp_name)

    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    setup_logger(save_dir,
                 distributed_rank=args.local_rank,
                 mode='w')
    logger.info('Args: {}'.format(args))

    model = build_local_model(config, device)
    model.head.nms = True

    model.cuda(args.local_rank)
    model.eval()

    ckpt_file = args.ckpt
    logger.info('loading checkpoint from {}'.format(ckpt_file))
    loc = 'cuda:{}'.format(args.local_rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    new_state_dict = {}
    for k, v in ckpt['model'].items():
        k = k.replace('module', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    logger.info('loaded checkpoint done.')

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    infer_shape = sum(config.test.augment.transform.image_max_range) // 2
    logger.info('Model Summary: {}'.format(get_model_info(model,
        (infer_shape, infer_shape))))

    model = build_ddp_model(model, local_rank=args.local_rank)
    if args.fuse:
        logger.info('\tFusing model...')
        model = fuse_model(model)
    # start evaluate
    output_folders = [None] * len(config.dataset.val_ann)

    if args.local_rank == 0 and config.miscs.output_dir:
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
                                  size_div=32)

    for output_folder, dataset_name, data_loader_val in zip(
            output_folders, config.dataset.val_ann, val_loader):
        inference(
            model,
            data_loader_val,
            dataset_name,
            iou_types=('bbox', ),
            box_only=False,
            device=device,
            output_folder=output_folder,
        )


if __name__ == '__main__':
    main()
