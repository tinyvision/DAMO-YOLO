# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import tensorrt as trt
from cuda import cuda
from damo.dataset.datasets.evaluation import evaluate
from damo.structures.bounding_box import BoxList
from damo.utils import postprocess
from damo.utils.timer import Timer

COCO_CLASSES = []
for i in range(80):
    COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def compute_on_dataset(config,
                       context,
                       data_loader,
                       device,
                       timer=None,
                       end2end=False):

    results_dict = {}
    cpu_device = torch.device('cpu')
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

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            images_np = images.tensors.numpy()
            input_batch = images_np.astype(np.float32)

            trt_out = []
            for output in outputs:
                trt_out.append(np.zeros(output['shape'], output['dtype']))

            def predict(batch):  # result gets copied into output
                # transfer input data to device
                cuda.cuMemcpyHtoD(inputs[0]['allocation'][1],
                                  np.ascontiguousarray(batch),
                                  int(inputs[0]['size']))
                # execute model
                context.execute_v2(allocations)
                # transfer predictions back
                for o in range(len(trt_out)):
                    cuda.cuMemcpyDtoH(trt_out[o], outputs[o]['allocation'][1],
                                      outputs[o]['size'])
                return trt_out

            pred_out = predict(input_batch)
            # trt with nms
            if end2end:
                nums = pred_out[0]
                boxes = pred_out[1]
                scores = pred_out[2]
                pred_classes = pred_out[3]
                batch_size = boxes.shape[0]
                output = [None for _ in range(batch_size)]
                for i in range(batch_size):
                    img_h, img_w = images.image_sizes[i]
                    boxlist = BoxList(torch.Tensor(boxes[i][:nums[i][0]]),
                                      (img_w, img_h),
                                      mode='xyxy')
                    boxlist.add_field(
                        'objectness',
                        torch.Tensor(np.ones_like(scores[i][:nums[i][0]])))
                    boxlist.add_field('scores',
                                      torch.Tensor(scores[i][:nums[i][0]]))
                    boxlist.add_field(
                        'labels',
                        torch.Tensor(pred_classes[i][:nums[i][0]] + 1))
                    output[i] = boxlist
            else:
                cls_scores = torch.Tensor(pred_out[0])
                bbox_preds = torch.Tensor(pred_out[1])
                output = postprocess(cls_scores, bbox_preds,
                                     config.model.head.num_classes,
                                     config.model.head.nms_conf_thre,
                                     config.model.head.nms_iou_thre,
                                     images)

            if timer:
                torch.cuda.synchronize()
                timer.toc()

            output = [o.to(cpu_device) if o is not None else o for o in output]
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, output)})
    return results_dict


def inference(
    config,
    context,
    data_loader,
    dataset_name,
    iou_types=('bbox', ),
    box_only=False,
    device='cuda',
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    end2end=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    dataset = data_loader.dataset
    logger.info('Start evaluation on {} dataset({} images).'.format(
        dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(config, context, data_loader, device,
                                     inference_timer, end2end)
    # convert to a list
    image_ids = list(sorted(predictions.keys()))
    predictions = [predictions[i] for i in image_ids]

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
