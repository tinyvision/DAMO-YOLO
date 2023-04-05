# Partial Quantization

The performance of DAMO-YOLO-S is seriously reduced from 46.8% to 33.6% after traditional PTQs, which is unacceptable. In order to solve this problem, we apply partial quantization. We quantified each layer of the model separately at the TRT level, analyzed each layer with precision as sensitivity, and then let the most sensitive layer to have full precision as a compromise.

With partial quantization, we finally reached 46.5% with a loss of only 0.3% in accuracy on DAMO-YOLO-S. Compared with the FP16 model, the partial quantization model accelerates by 20% when the batch size is 1, showing a good compromise between accuracy and latency.

DAMO-YOLO-T, DAMO-YOLO-M quantized model have been released, please feel free to use them.

## Prerequirements

TRT Version: 8.4.1.5

```python
pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com nvidia-pyindex
pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com pytorch_quantization
```

## Partial quantization

by specifying the layer to be quanted, we proceed partial quantization as follows, the calib weights, onnx files and trt files will be generated.

```python
# tiny
python tools/partial_quantization/partial_quant.py -f configs/damoyolo_tinynasL20_T.py -c damoyolo_tinynasL20_T.pth --batch_size 1 --img_size 640 --trt --trt_eval --model_type tiny
# small
python tools/partial_quantization/partial_quant.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640 --trt --trt_eval --model_type small
# medium
python tools/partial_quantization/partial_quant.py -f configs/damoyolo_tinynasL35_M.py -c damoyolo_tinynasL35_M.pth --batch_size 1 --img_size 640 --trt --trt_eval --model_type medium
```

## Latency Measurement

TRT model latency can be measured by trtexec.

```python
trtexec --avgRuns=1000 --workspace=1024 --loadEngine=damoyolo_tinynasL25_S_partial_quant_bs1.trt
```

## Performance

| Model           | Size        | Precision        |mAP_val(0.5:0.95) | T4 Latency bs=1 (ms) |
| :-------------- | ----------- | ----------- |:----------------------- | ---------------------------------------- |
| [**DAMOYOLO-T-partial**](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/quant_model/damoyolo_tinynasL20_T_partial_quant_bs1.trt)| 640 | INT8  | 42.7 | 2.39 |
| [**DAMOYOLO-S-FP16**](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/quant_model/damoyolo_tinynasL20_T_fp16_bs1.trt) | 640  | FP16 | 43.0  | 2.78 |
| [**DAMOYOLO-S-partial**](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/quant_model/damoyolo_tinynasL25_S_partial_quant_bs1.trt)| 640 | INT8  | 46.5 | 3.23 |
| [**DAMOYOLO-S-FP16**](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/quant_model/damoyolo_tinynasL25_S_fp16_bs1.trt) | 640  | FP16 | 46.8  | 3.83 |
| [**DAMOYOLO-M-partial**](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/quant_model/damoyolo_tinynasL35_M_partial_quant_bs1.trt)| 640 | INT8  | 49.5 | 4.57 |
| [**DAMOYOLO-M-FP16**](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/quant_model/damoyolo_tinynasL35_M_fp16_bs1.trt) | 640  | FP16 | 50.0  | 5.62 |

