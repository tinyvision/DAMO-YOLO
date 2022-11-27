English | [简体中文](README_cn.md)

<div align="center"><img src="assets/logo.png" width="1500"></div>

## Introduction
Welcome to **DAMO-YOLO**! It is a fast and accurate object detection method, which achieves a higher performance than state-of-the-art YOLO series. DAMO-YOLO is extend from YOLO but with some new techs, including Neural Architecture Search (NAS) backbones, efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. For more details, please refer to our report on Arxiv(coming soon). Moreover, here you can find not only powerful models, but also highly efficient training strategies and complete tools from training to deployment.

<div align="center"><img src="assets/curve.png" width="500"></div>

## Updates
- **[2022/11/27: We release  DAMO-YOLO v0.1.0!]**
  * Release DAMO-YOLO object detection models, including DAMO-YOLO-T, DAMO-YOLO-S and DAMO-YOLO-M.
  * Release model convert tools for esay deployment, surppots onnx and TensorRT-fp32, TensorRT-fp16.

## Web Demo
- [DAMO-YOLO-S](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary) is integrated into ModelScope. Try out the Web Demo.

## Model Zoo
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency T4<br>TRT-FP16-BS1| FLOPs<br>(G)| Params<br>(M)| Download |
| ------        |:---: | :---:     |:---:|:---: | :---: | :---:|
|[DAMO-YOLO-T](./configs/damoyolo_tinynasL20_T.py) | 640 | 43.0  | 2.78  | 18.1  | 8.5  | [link](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/damoyolo_tinynasL20_T.pth)  |
|[DAMO-YOLO-S](./configs/damoyolo_tinynasL25_S.py) | 640 | 46.8  | 3.83  | 37.8  | 16.3  |[link](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/damoyolo_tinynasL25_S.pth)  |
|[DAMO-YOLO-M](./configs/damoyolo_tinynasL35_M.py) | 640 | 50.0  | 5.62  | 61.8  | 28.2  | [link](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/damoyolo_tinynasL35_M.pth) |

- We report the mAP of models on COCO2017 validation set, with multi-class NMS.
- The latency in this table is measured without post-processing.

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install DAMO-YOLO.
```shell
git clone https://github.com/tinyvision/damo-yolo.git
cd DAMO-YOLO/
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython;
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table, e.g., damoyolo_tinynasL25_S.

Step2. Use -f(config filename) to specify your detector's config. For example:
```shell
python tools/demo.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth --path assets/dog.jpg
```
</details>


<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <DAMO-YOLO Home>
ln -s /path/to/your/coco ./datasets/coco
```

Step 2. Reproduce our results on COCO by specifying -f(config filename)
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py
```
</details>

<details>
<summary>Finetune on your data</summary>

Step1. Prepare your customize data in COCO format, and make sure the dataset name ends with coco. The dataset structure should be organized as follows:
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
Step2. Add the data directoy into damo/config/paths_catalog.py. Customize your config file based on default configs, e.g., damoyolo_TinynasL25_S.py. Don't forget to add pretrained model by config.train.finetune_path='./damoyolo_TinynasL25_S.pth' and specify the learning_rate/training_epochs/datasets and other hyperparameters according to your data.

Step3. Start finetuning:
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S_finetune.py
``` 

</details>



<details>
<summary>Evaluation</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth
```
</details>


## Deploy
<details>
<summary>Installation</summary>

Step1. Install ONNX.
```shell
pip install onnx==1.8.1
pip install onnxruntime==1.8.0
pip install onnx-simplifier==0.3.5
```
Step2. Install CUDA、CuDNN、TensorRT and pyCUDA

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
<summary>Model Convert</summary>

Step.1 convert torch model to onnx or trt engine, and the output file would be generated in ./deploy. end2end means to export trt with nms. trt_eval means to evaluate the exported trt engine on coco_val dataset after the export compelete.
```shell
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640 --trt --end2end --trt_eval
```

Step.2 trt engine evaluation on coco_val dataset. end2end means to using trt_with_nms to evaluation.
```shell
python tools/trt_eval.py -f configs/damoyolo_tinynasL25_S.py -trt deploy/damoyolo_tinynasL25_S_end2end.trt --batch_size 1 --img_size 640 --end2end
```

Step.3 trt engine inference demo and appoint test image by -p. end2end means to using trt_with_nms to inference.
```shell
python tools/trt_inference.py -f configs/damoyolo_tinynasL25_s.py -t deploy/damoyolo_tinynasL25_S_end2end_fp16_bs1.trt -p assets/dog.jpg --img_size 640 --end2end
```
</details>

## Cite DAMO-YOLO
If you use DAMO-YOLO in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{damoyolo,
   title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
   author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
   journal={arXiv preprint arXiv:22xx.xxxxx},
   year={2022},
 }
```
