#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import argparse
import sys

import onnx
import torch
from loguru import logger
from torch import nn

from damo.base_models.core.end2end import End2End
from damo.base_models.core.ops import RepConv, SiLU
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils.model_utils import get_model_info, replace_module
from tools.trt_eval import trt_inference

from tools.partial_quantization.utils import post_train_quant, load_quanted_model, execute_partial_quant, init_calib_data_loader

from pytorch_quantization import nn as quant_nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_parser():
    parser = argparse.ArgumentParser('damo converter deployment toolbox')
    # mode part
    parser.add_argument('--mode',
                        default='onnx',
                        type=str,
                        help='onnx, trt_16 or trt_32')
    # model part
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='expriment description file',
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt path')
    parser.add_argument('--trt',
                        action='store_true',
                        help='whether convert onnx into tensorrt')
    parser.add_argument(
        '--trt_type', type=str, default='fp32',
        help='one type of int8, fp16, fp32')
    parser.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='inference image batch nums')
    parser.add_argument('--img_size',
                        type=int,
                        default='640',
                        help='inference image shape')
    # onnx part
    parser.add_argument('--input',
                        default='images',
                        type=str,
                        help='input node name of onnx model')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output node name of onnx model')
    parser.add_argument('-o',
                        '--opset',
                        default=11,
                        type=int,
                        help='onnx opset version')
    parser.add_argument('--calib_weights',
                        type=str,
                        default=None,
                        help='calib weights')
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        help='quant model type(tiny, small, medium)')
    parser.add_argument('--sensitivity_file',
                        type=str,
                        default=None,
                        help='sensitivity file')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='export end2end onnx')
    parser.add_argument('--ort',
                        action='store_true',
                        help='export onnx for onnxruntime')
    parser.add_argument('--trt_eval',
                        action='store_true',
                        help='trt evaluation')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='iou threshold for NMS')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.05,
                        help='conf threshold for NMS')
    parser.add_argument('--device',
                        default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser



@logger.catch
def trt_export(onnx_path, batch_size, inference_h, inference_w):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger()
    engine_path = onnx_path.replace('.onnx', f'_bs{batch_size}.trt')

    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:

        logger.info('Loading ONNX file from path {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as model:
            logger.info('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                logger.info('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    logger.info(parser.get_error(error))

        builder.max_batch_size = batch_size
        logger.info('Building an engine.  This would take a while...')
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30
        
        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(line, text)
            )

        logger.info('generated trt engine named {}'.format(engine_path))
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        return engine_path


@logger.catch
def main():
    args = make_parser().parse_args()

    logger.info('args value: {}'.format(args))

    onnx_name = args.config_file.split('/')[-1].replace('.py', '_partial_quant.onnx')
    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')

    # init config
    config = parse_config(args.config_file)
    config.merge(args.opts)
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size

    # build model
    model = build_local_model(config, 'cuda')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.eval()
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    logger.info('loading checkpoint done.')
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    info = get_model_info(model, (args.img_size, args.img_size))
    logger.info(info)

    # decouple postprocess
    model.head.nms = False

    # 1. do post training quantization
    if args.calib_weights is None:
        calib_data_loader = init_calib_data_loader(config)
        ptq_model = post_train_quant(model, calib_data_loader, 1000, device)
        torch.save({'model': ptq_model}, args.ckpt.replace('.pth', '_calib.pth'))
    else:
        ptq_model = load_quanted_model(model, args.calib_weights, device)

    # 2. load sensitivity data
    all_ops = list()
    for k, m in ptq_model.named_modules():
        if isinstance(m, quant_nn.QuantConv2d) or \
           isinstance(m, quant_nn.QuantConvTranspose2d) or \
           isinstance(m, quant_nn.MaxPool2d):
            all_ops.append((k))

    quant_model = args.model_type
    if quant_model == 'tiny':
        backbone_inds = list(range(24))
        neck_inds = []
        head_inds = list(range(74, 80))
    elif quant_model == 'small':
        backbone_inds = list(range(30))
        neck_inds = list(range(30,31)) + list(range(32,40)) + list(range(40,41)) + list(range(42, 49)) + list(range(50,51)) + list(range(52, 59)) + list(range(60, 61)) + list(range(62, 69)) + list(range(70, 71)) + list(range(72, 79))
        head_inds = list(range(80, 86))
    elif quant_model == 'medium':
        backbone_inds = list(range(5)) + list(range(6, 15)) + list(range(16, 33)) + list(range(34, 46)) + list(range(47, 48))
        neck_inds = []
        head_inds = list(range(108, 114))
    else:
        raise ValueError("unsupported model type in requested schema(tiny, small, medium)")

    all_inds = backbone_inds + neck_inds + head_inds

    quantable_sensitivity = [all_ops[x] for x in all_inds]
    ops_to_quant = [qops for qops in quantable_sensitivity]        

    # 3. only quantize ops in quantable_ops list
    execute_partial_quant(ptq_model, ops_to_quant=ops_to_quant)


    # 4. ONNX export
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    _ = ptq_model(dummy_input)
    torch.onnx._export(
        ptq_model,
        dummy_input,
        onnx_name,
        verbose=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        opset_version=13,
    )
    onnx_model = onnx.load(onnx_name)        # Fix output shape
    try:
        import onnxsim
        logger.info('Starting to simplify ONNX...')
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'check failed'
    except Exception as e:
        logger.info(f'simplify failed: {e}')
    onnx.save(onnx_model, onnx_name)
    logger.info('generated onnx model named {}'.format(onnx_name))

    # 5. export trt
    if args.trt:
        trt_name = trt_export(onnx_name, args.batch_size, args.img_size, args.img_size)
        # 6. trt eval
        if args.trt_eval:
            logger.info('start trt inference on coco validataion dataset')
            trt_inference(config, trt_name, args.img_size, args.batch_size,
                          args.conf_thres, args.iou_thres, args.end2end)


if __name__ == '__main__':
    main()
