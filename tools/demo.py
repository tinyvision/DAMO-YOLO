# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import get_model_info, vis, postprocess
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from damo.structures.bounding_box import BoxList

IMAGES=['png', 'jpg']
VIDEOS=['mp4', 'avi']


class Infer():
    def __init__(self, config, infer_size=[640,640], device='cuda', output_dir='./', ckpt=None, end2end=False):

        self.ckpt_path = ckpt
        suffix = ckpt.split('.')[-1]
        if suffix == 'onnx':
            self.engine_type = 'onnx'
        elif suffix == 'trt':
            self.engine_type = 'tensorRT'
        elif suffix in ['pt', 'pth']:
            self.engine_type = 'torch'
        self.end2end = end2end # only work with tensorRT engine
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if torch.cuda.is_available() and device=='cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if "class_names" in config.dataset:
            self.class_names = config.dataset.class_names
        else:
            self.class_names = []
            for i in range(config.model.head.num_classes):
                self.class_names.append(str(i))
            self.class_names = tuple(self.class_names)

        self.infer_size = infer_size
        config.dataset.size_divisibility = 0
        self.config = config
        self.model = self._build_engine(self.config, self.engine_type)

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        assert h<=target_size[0] and w<=target_size[1]
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = torch.zeros(*target_size)
        pad_imgs[:, :c, :h, :w].copy_(img)

        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]

        return ImageList(pad_imgs, img_sizes, pad_sizes)


    def _build_engine(self, config, engine_type):

        print(f'Inference with {engine_type} engine!')
        if engine_type == 'torch':
            model = build_local_model(config, self.device)
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt['model'], strict=True)
            for layer in model.modules():
                if isinstance(layer, RepConv):
                    layer.switch_to_deploy()
            model.eval()
        elif engine_type == 'tensorRT':
            model = self.build_tensorRT_engine(self.ckpt_path)
        elif engine_type == 'onnx':
            model, self.input_name, self.infer_size, _, _ = self.build_onnx_engine(self.ckpt_path)
        else:
            NotImplementedError(f'{engine_type} is not supported yet! Please use one of [onnx, torch, tensorRT]')

        return model

    def build_tensorRT_engine(self, trt_path):

        import tensorrt as trt
        from cuda import cuda
        loggert = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(loggert, '')
        runtime = trt.Runtime(loggert)
        with open(trt_path, 'rb') as t:
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

        return predict




    def build_onnx_engine(self, onnx_path):

        import onnxruntime

        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        out_names = []
        out_shapes = []
        for idx in range(len(session.get_outputs())):
            out_names.append(session.get_outputs()[idx].name)
            out_shapes.append(session.get_outputs()[idx].shape)
        return session, input_name, input_shape[2:], out_names, out_shapes



    def preprocess(self, origin_img):

        img = transform_img(origin_img, 0,
                            **self.config.test.augment.transform,
                            infer_size=self.infer_size)
        # img is a image_list
        oh, ow, _  = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size)

        img = img.to(self.device)
        return img, (ow, oh)

    def postprocess(self, preds, image, origin_shape=None):

        if self.engine_type == 'torch':
            output = preds

        elif self.engine_type == 'onnx':
            scores = torch.Tensor(preds[0])
            bboxes = torch.Tensor(preds[1])
            output = postprocess(scores, bboxes,
                self.config.model.head.num_classes,
                self.config.model.head.nms_conf_thre,
                self.config.model.head.nms_iou_thre,
                image)
        elif self.engine_type == 'tensorRT':
            if self.end2end:
                nums = preds[0]
                boxes = preds[1]
                scores = preds[2]
                pred_classes = preds[3]
                batch_size = boxes.shape[0]
                output = [None for _ in range(batch_size)]
                for i in range(batch_size):
                    img_h, img_w = image.image_sizes[i]
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
                cls_scores = torch.Tensor(preds[0])
                bbox_preds = torch.Tensor(preds[1])
                output = postprocess(cls_scores, bbox_preds,
                             self.config.model.head.num_classes,
                             self.config.model.head.nms_conf_thre,
                             self.config.model.head.nms_iou_thre, image)

        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field('scores')
        cls_inds = output.get_field('labels')

        return bboxes,  scores, cls_inds


    def forward(self, origin_image):

        image, origin_shape = self.preprocess(origin_image)

        if self.engine_type == 'torch':
            output = self.model(image)

        elif self.engine_type == 'onnx':
            image_np = np.asarray(image.tensors.cpu())
            output = self.model.run(None, {self.input_name: image_np})

        elif self.engine_type == 'tensorRT':
            image_np = np.asarray(image.tensors.cpu()).astype(np.float32)
            output = self.model(image_np)

        bboxes, scores, cls_inds = self.postprocess(output, image, origin_shape=origin_shape)

        return bboxes, scores, cls_inds

    def visualize(self, image, bboxes, scores, cls_inds, conf, save_name='vis.jpg', save_result=True):
        vis_img = vis(image, bboxes, scores, cls_inds, conf, self.class_names)
        if save_result:
            save_path = os.path.join(self.output_dir, save_name)
            print(f"save visualization results at {save_path}")
            cv2.imwrite(save_path, vis_img[:, :, ::-1])
        return vis_img


def make_parser():
    parser = argparse.ArgumentParser('DAMO-YOLO Demo')

    parser.add_argument('input_type',
                        default='image',
                        help="input type, support [image, video, camera]")
    parser.add_argument('-f',
                        '--config_file',
                        default=None,
                        type=str,
                        help='pls input your config file',)
    parser.add_argument('-p',
                        '--path',
                        default='./assets/dog.jpg',
                        type=str,
                        help='path to image or video')
    parser.add_argument('--camid',
                        type=int,
                        default=0,
                        help='camera id, necessary when input_type is camera')
    parser.add_argument('--engine',
                        default=None,
                        type=str,
                        help='engine for inference')
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help='device used to inference')
    parser.add_argument('--output_dir',
                        default='./demo',
                        type=str,
                        help='where to save inference results')
    parser.add_argument('--conf',
                        default=0.6,
                        type=float,
                        help='conf of visualization')
    parser.add_argument('--infer_size',
                        nargs='+',
                        type=int,
                        help='test img size')
    parser.add_argument('--end2end',
                        action='store_true',
                        help='trt engine with nms')
    parser.add_argument('--save_result',
                        default=True,
                        type=bool,
                        help='whether save visualization results')


    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    config = parse_config(args.config_file)
    input_type = args.input_type

    infer_engine = Infer(config, infer_size=args.infer_size, device=args.device,
        output_dir=args.output_dir, ckpt=args.engine, end2end=args.end2end)

    if input_type == 'image':
        origin_img = np.asarray(Image.open(args.path).convert('RGB'))
        bboxes, scores, cls_inds = infer_engine.forward(origin_img)
        vis_res = infer_engine.visualize(origin_img, bboxes, scores, cls_inds, conf=args.conf, save_name=os.path.basename(args.path), save_result=args.save_result)
        if not args.save_result:
            cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
            cv2.imshow("DAMO-YOLO", vis_res)

    elif input_type == 'video' or input_type == 'camera':
        cap = cv2.VideoCapture(args.path if input_type == 'video' else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        if args.save_result:
            save_path = os.path.join(args.output_dir, os.path.basename(args.path))
            print(f'inference result will be saved at {save_path}')
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (int(width), int(height)))
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                bboxes, scores, cls_inds = infer_engine.forward(frame)
                result_frame = infer_engine.visualize(frame, bboxes, scores, cls_inds, conf=args.conf, save_result=False)
                if args.save_result:
                    vid_writer.write(result_frame)
                else:
                    cv2.namedWindow("DAMO-YOLO", cv2.WINDOW_NORMAL)
                    cv2.imshow("DAMO-YOLO", result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break



if __name__ == '__main__':
    main()
