# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import datetime
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from damo.apis.detector_inference import inference
from damo.base_models.losses.distill_loss import FeatureLoss
from damo.dataset import build_dataloader, build_dataset
from damo.detectors.detector import build_ddp_model, build_local_model
from damo.utils import (MeterBuffer, get_model_info, get_rank, gpu_mem_usage,
                        save_checkpoint, setup_logger, synchronize)

from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
NORMS = (GroupNorm, LayerNorm, _BatchNorm)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class cosine_scheduler:
    def __init__(self,
                 base_lr_per_img,
                 batch_size,
                 min_lr_ratio,
                 total_iters,
                 no_aug_iters,
                 warmup_iters,
                 warmup_start_lr=0):

        self.base_lr = base_lr_per_img * batch_size
        self.final_lr = self.base_lr * min_lr_ratio
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr
        self.total_iters = total_iters
        self.no_aug_iters = no_aug_iters

    def get_lr(self, iters):

        if iters < self.warmup_iters:
            lr = (self.base_lr - self.warmup_start_lr) * pow(
                iters / float(self.warmup_iters), 2) + self.warmup_start_lr
        elif iters >= self.total_iters - self.no_aug_iters:
            lr = self.final_lr
        else:
            lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
                1.0 + math.cos(math.pi * (iters - self.warmup_iters) /
                               (self.total_iters - self.warmup_iters -
                                self.no_aug_iters)))
        return lr


class ema_model:
    def __init__(self, student, ema_momentum):

        self.model = deepcopy(student).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.ema_scheduler = lambda x: ema_momentum * (1 - math.exp(-x / 2000))

    def update(self, iters, student):

        student = student.module.state_dict()
        with torch.no_grad():
            momentum = self.ema_scheduler(iters)
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    param *= momentum
                    param += (1.0 - momentum) * student[name].detach()


class Trainer:
    def __init__(self, cfg, args, tea_cfg=None, is_train=True):
        self.cfg = cfg
        self.tea_cfg = tea_cfg
        self.args = args
        self.output_dir = cfg.miscs.output_dir
        self.exp_name = cfg.miscs.exp_name
        self.device = 'cuda'

        # set_seed(cfg.miscs.seed)
        # metric record
        self.meter = MeterBuffer(window_size=cfg.miscs.print_interval_iters)
        self.file_name = os.path.join(cfg.miscs.output_dir, cfg.miscs.exp_name)

        # setup logger
        if get_rank() == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=get_rank(),
            mode='w',
            )

        # logger
        logger.info('args info: {}'.format(self.args))
        logger.info('cfg value:\n{}'.format(self.cfg))

        # build model
        self.model = build_local_model(self.cfg, self.device)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        logger.info('model:', self.model)

        if tea_cfg is not None:
            self.distill = True
            self.grad_clip = 30
            self.tea_model = build_local_model(self.tea_cfg, self.device)
            self.tea_model.eval()
            tea_ckpt = torch.load(args.tea_ckpt, map_location=self.device)
            #self.tea_model.load_state_dict(tea_ckpt['model'], strict=True)
            self.tea_model.load_pretrain_detector(args.tea_ckpt)
            self.feature_loss = FeatureLoss(self.model.neck.out_channels,
                                            self.tea_model.neck.out_channels,
                                            distiller='cwd').to(self.device)
            self.optimizer = self.build_optimizer((self.model, self.feature_loss),
                cfg.train.optimizer)
        else:
            self.distill = False
            self.grad_clip = None

            self.optimizer = self.build_optimizer(self.model,
                cfg.train.optimizer)
        # resume model
        if self.cfg.train.finetune_path is not None:
            self.model.load_pretrain_detector(self.cfg.train.finetune_path)
            self.epoch = 0
            self.start_epoch = 0
        elif self.cfg.train.resume_path is not None:
            resume_epoch = self.resume_model(self.cfg.train.resume_path,
                                             load_optimizer=True)
            self.epoch = resume_epoch
            self.start_epoch = resume_epoch
            logger.info('Resume Training from Epoch: {}'.format(self.epoch))
        else:
            self.epoch = 0
            self.start_epoch = 0
            logger.info('Start Training...')

        if self.cfg.train.ema:
            logger.info(
                'Enable ema model! Ema model will be evaluated and saved.')
            self.ema_model = ema_model(self.model, cfg.train.ema_momentum)
        else:
            self.ema_model = None

        # dataloader
        self.train_loader, self.val_loader, iters = self.get_data_loader(cfg)

        # setup iters according epochs and iters_per_epoch
        self.setup_iters(iters, self.start_epoch, cfg.train.total_epochs,
                         cfg.train.warmup_epochs, cfg.train.no_aug_epochs,
                         cfg.miscs.eval_interval_epochs,
                         cfg.miscs.ckpt_interval_epochs,
                         cfg.miscs.print_interval_iters)

        self.lr_scheduler = cosine_scheduler(
            cfg.train.base_lr_per_img, cfg.train.batch_size,
            cfg.train.min_lr_ratio, self.total_iters, self.no_aug_iters,
            self.warmup_iters, cfg.train.warmup_start_lr)

        self.mosaic_mixup = 'mosaic_mixup' in cfg.train.augment

    def get_data_loader(self, cfg):

        train_dataset = build_dataset(
            cfg,
            cfg.dataset.train_ann,
            is_train=True,
            mosaic_mixup=cfg.train.augment.mosaic_mixup)
        val_dataset = build_dataset(cfg, cfg.dataset.val_ann, is_train=False)

        iters_per_epoch = math.ceil(
            len(train_dataset[0]) /
            cfg.train.batch_size)  # train_dataset is a list, however,

        train_loader = build_dataloader(train_dataset,
                                        cfg.train.augment,
                                        batch_size=cfg.train.batch_size,
                                        start_epoch=self.start_epoch,
                                        total_epochs=cfg.train.total_epochs,
                                        num_workers=cfg.miscs.num_workers,
                                        is_train=True,
                                        size_div=32)

        val_loader = build_dataloader(val_dataset,
                                      cfg.test.augment,
                                      batch_size=cfg.test.batch_size,
                                      num_workers=cfg.miscs.num_workers,
                                      is_train=False,
                                      size_div=32)

        return train_loader, val_loader, iters_per_epoch

    def setup_iters(self, iters_per_epoch, start_epoch, total_epochs,
                    warmup_epochs, no_aug_epochs, eval_interval_epochs,
                    ckpt_interval_epochs, print_interval_iters):
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch
        self.start_iter = start_epoch * iters_per_epoch
        self.total_iters = total_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.no_aug_iters = no_aug_epochs * iters_per_epoch
        self.no_aug = self.start_iter >= self.total_iters - self.no_aug_iters
        self.eval_interval_iters = eval_interval_epochs * iters_per_epoch
        self.ckpt_interval_iters = ckpt_interval_epochs * iters_per_epoch
        self.print_interval_iters = print_interval_iters

    def build_optimizer(self, models, cfg, exp_module=None):
        if not isinstance(models, (tuple, list)):
            models = (models, )

        param_dict = {}
        base_wd = cfg.get('weight_decay', None)
        optimizer_name = cfg.pop('name')
        optim_cls = getattr(torch.optim, optimizer_name)
        for model in models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert param not in param_dict
                param_dict[param] = {"name": name}

            # weight decay of bn is always 0.
            for name, m in model.named_modules():
                if isinstance(m, NORMS):
                    if hasattr(m, "bias") and m.bias is not None:
                        param_dict[m.bias].update({"weight_decay": 0})
                    param_dict[m.weight].update({"weight_decay": 0})

            # weight decay of bias is always 0.
            for name, m in model.named_modules():
                if hasattr(m, "bias") and m.bias is not None:
                    param_dict[m.bias].update({"weight_decay": 0})
            param_groups = []
        for p, pconfig in param_dict.items():
            name = pconfig.pop("name", None)
            param_groups += [{"params": p, **pconfig}]


        optimizer = optim_cls(param_groups, **cfg)

        return optimizer

    def train(self, local_rank):

        infer_shape = sum(self.cfg.test.augment.transform.image_max_range) // 2
        logger.info('Model Summary: {}'.format(
            get_model_info(self.model, (infer_shape, infer_shape))))

        # distributed model init
        self.model = build_ddp_model(self.model, local_rank)
        logger.info('Model: {}'.format(self.model))

        logger.info('Training start...')

        # ----------- start training ------------------------- #
        self.model.train()
        iter_start_time = time.time()
        iter_end_time = time.time()
        for data_iter, (inps, targets, ids) in enumerate(self.train_loader):
            cur_iter = self.start_iter + data_iter

            lr = self.lr_scheduler.get_lr(cur_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            inps = inps.to(self.device)  # ImageList: tensors, img_size
            targets = [target.to(self.device)
                       for target in targets]  # BoxList: bbox, num_boxes ...

            model_start_time = time.time()

            if self.distill:
                outputs, fpn_outs = self.model(inps, targets, stu=True)
                loss = outputs['total_loss']
                with torch.no_grad():
                    fpn_outs_tea = self.tea_model(inps, targets, tea=True)
                distill_weight = (
                    (1 - math.cos(cur_iter * math.pi / len(self.train_loader)))
                    / 2) * (0.1 - 1) + 1

                distill_loss = distill_weight * self.feature_loss(
                    fpn_outs, fpn_outs_tea)
                loss = loss + distill_loss
                outputs['distill_loss'] = distill_loss

            else:

                outputs = self.model(inps, targets)
                loss = outputs['total_loss']

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.grad_clip,
                                         norm_type=2)  # for stable training

            self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update(cur_iter, self.model)

            iter_start_time = iter_end_time
            iter_end_time = time.time()

            outputs_array = {_name: _v.item() for _name, _v in outputs.items()}
            self.meter.update(
                iter_time=iter_end_time - iter_start_time,
                model_time=iter_end_time - model_start_time,
                lr=lr,
                **outputs_array,
            )

            if cur_iter + 1 > self.total_iters - self.no_aug_iters:
                if self.mosaic_mixup:
                    logger.info('--->turn OFF mosaic aug now!')
                    self.train_loader.batch_sampler.set_mosaic(False)
                    self.eval_interval_iters = self.iters_per_epoch
                    self.ckpt_interval_iters = self.iters_per_epoch
                    self.mosaic_mixup = False

            # log needed information
            if (cur_iter + 1) % self.print_interval_iters == 0:
                left_iters = self.total_iters - (cur_iter + 1)
                eta_seconds = self.meter['iter_time'].global_avg * left_iters
                eta_str = 'ETA: {}'.format(
                    datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = 'epoch: {}/{}, iter: {}/{}'.format(
                    self.epoch + 1, self.total_epochs,
                    (cur_iter + 1) % self.iters_per_epoch,
                    self.iters_per_epoch)
                loss_meter = self.meter.get_filtered_meter('loss')
                loss_str = ', '.join([
                    '{}: {:.1f}'.format(k, v.avg)
                    for k, v in loss_meter.items()
                ])

                time_meter = self.meter.get_filtered_meter('time')
                time_str = ', '.join([
                    '{}: {:.3f}s'.format(k, v.avg)
                    for k, v in time_meter.items()
                ])

                logger.info('{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}'.format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter['lr'].latest,
                ) + (', size: ({:d}, {:d}), {}'.format(
                    inps.tensors.shape[2], inps.tensors.shape[3], eta_str)))
                self.meter.clear_meters()

            if (cur_iter + 1) % self.ckpt_interval_iters == 0:
                self.save_ckpt('epoch_%d' % (self.epoch + 1),
                               local_rank=local_rank)

            if (cur_iter + 1) % self.eval_interval_iters == 0:
                time.sleep(0.003)
                self.evaluate(local_rank, self.cfg.dataset.val_ann)
                self.model.train()
            synchronize()

            if (cur_iter + 1) % self.iters_per_epoch == 0:
                self.epoch = self.epoch + 1

        self.save_ckpt(ckpt_name='latest', local_rank=local_rank)

    def save_ckpt(self, ckpt_name, local_rank, update_best_ckpt=False):
        if local_rank == 0:
            if self.ema_model is not None:
                save_model = self.ema_model.model
            else:
                save_model = self.model.module
            logger.info('Save weights to {}'.format(self.file_name))
            ckpt_state = {
                'epoch':
                self.epoch + 1,
                'model':
                save_model.state_dict(),
                'optimizer':
                self.optimizer.state_dict(),
                'feature_loss':
                self.feature_loss.state_dict() if self.distill else None,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def resume_model(self, resume_path, load_optimizer=False):
        ckpt_file_path = resume_path
        ckpt = torch.load(ckpt_file_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.distill:
                self.feature_loss.load_state_dict(ckpt['feature_loss'])
            resume_epoch = ckpt['epoch']
            return resume_epoch

    def evaluate(self, local_rank, val_ann):
        assert len(self.val_loader) == len(val_ann)
        if self.ema_model is not None:
            evalmodel = self.ema_model.model
        else:
            evalmodel = self.model
            if isinstance(evalmodel, DDP):
                evalmodel = evalmodel.module

        output_folders = [None] * len(val_ann)
        for idx, dataset_name in enumerate(val_ann):
            output_folder = os.path.join(self.output_dir, self.exp_name,
                                         'inference', dataset_name)
            if local_rank == 0:
                mkdir(output_folder)
            output_folders[idx] = output_folder

        for output_folder, dataset_name, data_loader_val in zip(
                output_folders, val_ann, self.val_loader):
            inference(
                evalmodel,
                data_loader_val,
                dataset_name,
                device=self.device,
                output_folder=output_folder,
            )
