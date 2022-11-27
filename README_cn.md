[English](README.md) | 简体中文

<div align="center"><img src="assets/logo.png" width="1500"></div>

## 简介
欢迎来到**DAMO-YOLO**！DAMO-YOLO是一个兼顾速度与精度的目标检测框架，其效果超越了目前的一众YOLO系列方法，在实现SOTA的同时，保持了很高的推理速度。DAMO-YOLO是在YOLO框架基础上引入了一系列新技术，对整个检测框架进行了大幅的修改。具体包括：基于NAS搜索的新检测backbone结构，更深的neck结构，精简的head结构，以及引入蒸馏技术实现效果的进一步提升。具体细节可以参考我们的技术报告（即将发布）。模型之外，DAMO-YOLO还提供高效的训练策略以及便捷易用的部署工具，帮助您快速解决工业落地中的实际问题！

<div align="center"><img src="assets/curve.png" width="500"></div>

## 更新日志
-  **[2022/11/27: DAMO-YOLO v0.1.0开源!]**
    * 开源DAMO-YOLO-T, DAMO-YOLO-S和DAMO-YOLO-M模型。
    * 开源模型转换工具，支持onnx导出以及TensorRT-fp32、TensorRT-fp16模型转换。

## 线上Demo
- 线上Demo已整合至ModelScope，快去[DAMO-YOLO-S](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary) 体验一下吧！

## 模型库
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency T4<br>TRT-FP16-BS1| FLOPs<br>(G)| Params<br>(M)| Download|
| ------        |:---: | :---:     |:---:|:---: | :---: |:---:|
|[DAMO-YOLO-T](./configs/damoyolo_tinynasL20_T.py) | 640 | 43.0  | 2.78  | 18.1  | 8.5  | [link](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/damoyolo_tinynasL20_T.pth)|
|[DAMO-YOLO-S](./configs/damoyolo_tinynasL25_S.py) | 640 | 46.8  | 3.83  | 37.8  | 16.3  | [link](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/damoyolo_tinynasL25_S.pth) |
|[DAMO-YOLO-M](./configs/damoyolo_tinynasL35_M.py) | 640 | 50.0  | 5.62  | 61.8  | 28.2  | [link](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/damoyolo_tinynasL35_M.pth)|


- 上表中汇报的是COCO2017 val集上的结果, 测试时使用multi-class NMS。
- 其中latency中不包括后处理时间。

## 快速上手

<details>
<summary>安装</summary>

步骤一.  安装DAMO-YOLO.
```shell
git clone https://github.com/tinyvision/DAMO-YOLO.git
cd DAMO-YOLO/
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
步骤二. 安装[pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython;
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
</details>

<details>
<summary>Demo</summary>

步骤一. 从模型库中下载训练好的模型，例如damoyolo_tinynasL25_S.

步骤二. 执行命令时用-f选项指定配置(config)文件。例如:
```shell
python tools/demo.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth --path assets/dog.jpg
```
</details>

<details>
<summary>从头开始，复现COCO上的精度</summary>

步骤一. 准备好COCO数据集,推荐将coco数据软链接到datasets目录下。
```shell
cd <DAMO-YOLO Home>
ln -s /path/to/your/coco ./datasets/coco
```

步骤二. 在COCO数据上进行训练，使用-f选项指定配置(config)文件。
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py
```
</details>

<details>
<summary>在自定义数据上微调模型</summary>

Step1. 将您的自定义数据转换成COCO格式，并且将数据集路径添加到damo/config/paths_catalog.py，确保您的自定义数据集名称以"coco"结尾。数据的目录组织结构如下: 
```
├── Custom_coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── train2017
│   ├── val2017
│   ├── LICENSE
│   ├── README.txt
```
Step2. 在配置文件中加入预训练模型路径，例如: config.train.finetune_path='./damoyolo_tinynasL25_S.pth'，最后根据您的自定义数据的数据量和数据特点，修改配置文件中的learning_rate/training epochs/datasets和其他必要超参。 

Step3. 开始微调训练:
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S_finetune.py
``` 
</details>


<details>
<summary>在COCO val上测评训练好的模型</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth
```
</details>

## 部署

<details>
<summary>安装依赖项</summary>

步骤1. 安装 ONNX.
```shell
pip install onnx==1.8.1
pip install onnxruntime==1.8.0
pip install onnx-simplifier==0.3.5
```
步骤2. 安装 CUDA、CuDNN、TensorRT and pyCUDA

2.1 CUDA
```shell
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
source ~/.bashrc
```
2.2 CuDNN
```shell
sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
2.3 TensorRT
```shell
cd TensorRT-7.2.1.6/python
pip install tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-7.2.1.6/lib
```
2.4 pycuda
```shell
pip install pycuda==2022.1
```
</details>


<details>
<summary>模型转换</summary>

步骤一：将torch模型转换成onnx或者TensorRT推理引擎。具体使用方法如下：
```shell
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640 --trt --end2end --trt_eval
```
其中--end2end表示在导出的onnx或者TensorRT引擎中集成NMS模块，--trt_eval表示在TensorRT导出完成后即在coco2017 val上进行精度验证。

已经完成TensorRT导出的模型也可由如下指令在coco2017 val上进行精度验证。--end2end表示待测试的TensorRT引擎包含NMS组件。

```shell
python tools/trt_eval.py -f configs/damoyolo_tinynasL25_S.py -trt deploy/damoyolo_tinynasL25_S_end2end.trt --batch_size 1 --img_size 640 --end2end
```

步骤二：使用已经导出的TensorRT引擎进行目标检测。
```shell
python tools/trt_inference.py -f configs/damoyolo_tinynasL25_s.py -t deploy/damoyolo_tinynasL25_S_end2end_fp16_bs1.trt -p assets/dog.jpg --img_size 640 --end2end
```
</details>

## 引用

```latex
 @article{damoyolo,
   title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
   author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
   journal={arXiv preprint arXiv:22xx.xxxxx},
   year={2022},
 }
```
