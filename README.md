English | [简体中文](README_cn.md)

<div align="center"><img src="assets/logo.png" width="1500">

![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/license-Apache-000000.svg)
![Contributing](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
[![README-cn](https://shields.io/badge/README-%E4%B8%AD%E6%96%87-blue)](README_cn.md)
[![ThirdParty](https://img.shields.io/badge/ThirdParty--Resources-brightgreen)](#third-parry-resources)
[![IndustryModels](https://img.shields.io/badge/Industry--Models-orange)](#industry-application-models)

</div>

## Introduction
<div align="center"><img src="assets/overview.gif" width="1500"></div>

Welcome to **DAMO-YOLO**! It is a fast and accurate object detection method, which is developed by TinyML Team from Alibaba DAMO Data Analytics and Intelligence Lab. And it achieves a higher performance than state-of-the-art YOLO series. DAMO-YOLO is extend from YOLO but with some new techs, including Neural Architecture Search (NAS) backbones, efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. For more details, please refer to our [Arxiv Report](https://arxiv.org/abs/2211.15444). Moreover, here you can find not only powerful models, but also highly efficient training strategies and complete tools from training to deployment.

<div align="center"><img src="assets/curve.png" width="1200"></div>

## Updates
- **[2023/04/12: We release DAMO-YOLO v0.3.1!] ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png)**
    * Add 701-categories DAMO-YOLO-S model, which cover more application scenarios and serve as high-quality pre-training model to improve the performance of downstream tasks。
    * Upgrade the DAMO-YOLO-Nano series model, which achieves 32.3/38.2/40.5 mAP with only 1.56/3.69/6.04 Flops, and runs in real-time at 4.08/5.05/6.69ms using Intel-CPU. 
    * Add DAMO-YOLO-L model, which achieves 51.9 mAP with 7.95ms latency using T4-GPU.
- **[2023/03/13: We release DAMO-YOLO v0.3.0!]**
    * Release DAMO-YOLO-Nano, which achieves 35.1 mAP with only 3.02GFlops.
    * Upgrade the optimizer builder, edits the optimizer config, you are able to use any optimizer supported by Pytorch.
    * Upgrade the data loading pipeline and training parameters, leading to significant improvements of DAMO-YOLO models, e.g., the mAP of DAMO-YOLO-T/S/M increased from 43.0/46.8/50.0 to 43.6/47.7/50.2 respectively. 
- **[2023/02/15: Baseline for The 3rd Anti-UAV Challenge.]**
    * Welcome to join [the 3rd Anti-UAV Challenge](https://anti-uav.github.io/Evaluate/) on CVPR2023. The Challenge provides baseline models trained by DAMO-YOLO, which can be found on [DamoYolo_Anti-UAV-23_S](https://modelscope.cn/models/damo/cv_tinynas_uav-detection_damoyolo/summary).
- **[2023/01/07: We release DAMO-YOLO v0.2.1!]**
    * Add [TensorRT Int8 Quantization Tutorial](./tools/partial_quantization/README.md), achieves 19% speed up with only 0.3% accuracy loss.
    * Add [general demo tools](#quick-start), support TensorRT/Onnx/Torch based vidoe/image/camera inference.
    * Add more [industry application models](#industry-application-models), including [human detection](https://www.modelscope.cn/models/damo/cv_tinynas_human-detection_damoyolo/summary), [helmet detection](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_safety-helmet/summary), [facemask detection](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_facemask/summary) and [cigarette detection](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_cigarette/summary).
    * Add [third-party resources](#third-party-resources), including [DAMO-YOLO Code Interpretation](https://blog.csdn.net/jyyqqq/article/details/128419143), [Practical Example for Finetuning on Custom Dataset](https://blog.csdn.net/Cwhgn/article/details/128447380?spm=1001.2014.3001.5501). 
- **[2022/12/15: We release  DAMO-YOLO v0.1.1!]**
  * Add a detailed [Custom Dataset Finetune Tutorial](./assets/CustomDatasetTutorial.md).
  * The stuck problem caused by no-label data (*e.g.*, [ISSUE#30](https://github.com/tinyvision/DAMO-YOLO/issues/30)) is solved. Feel free to contact us, we are 24h stand by.

## Web Demo
- [DAMO-YOLO-T](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo-t/summary), [DAMO-YOLO-S](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary), [DAMO-YOLO-M](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo-m/summary) is integrated into ModelScope. Training is supported on [ModelScope](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary) now! **Come and try DAMO-YOLO with free GPU resources provided by ModelScope.** 

## Model Zoo
### General Models

|Model |size |mAP<sup>val<br>0.5:0.95 | Latency T4<br>TRT-FP16-BS1| FLOPs<br>(G)| Params<br>(M)| AliYun Download | Google Download|
| ------        |:---: | :---:     |:---:|:---: | :---: | :---:| :---:|
|[DAMO-YOLO-T](./configs/damoyolo_tinynasL20_T.py) | 640 | 42.0  | 2.78  | 18.1  | 8.5  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL20_T_420.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL20_T_420.onnx)|--|
|[DAMO-YOLO-T*](./configs/damoyolo_tinynasL20_T.py) | 640 | 43.6  | 2.78  | 18.1  | 8.5  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL20_T_436.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL20_T_436.onnx)|--|
|[DAMO-YOLO-S](./configs/damoyolo_tinynasL25_S.py) | 640 | 46.0  | 3.83  | 37.8  | 16.3  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL25_S_460.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL25_S_460.onnx)|--|
|[DAMO-YOLO-S*](./configs/damoyolo_tinynasL25_S.py) | 640 | 47.7  | 3.83  | 37.8  | 16.3 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL25_S_477.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL25_S_477.onnx) |--|
|[DAMO-YOLO-M](./configs/damoyolo_tinynasL35_M.py) | 640 | 49.2  | 5.62  | 61.8  | 28.2  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL35_M_492.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL35_M_492.onnx)|--|
|[DAMO-YOLO-M*](./configs/damoyolo_tinynasL35_M.py) | 640 | 50.2  | 5.62  | 61.8  | 28.2 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL35_M_502.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL35_M_502.onnx)|--|
|[DAMO-YOLO-L](./configs/damoyolo_tinynasL45_L.py) | 640 | 50.8  | 7.95  | 97.3  | 42.1  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL45_L_508.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL45_L_508.onnx)|--|
|[DAMO-YOLO-L*](./configs/damoyolo_tinynasL45_L.py) | 640 | 51.9  | 7.95  | 97.3  | 42.1  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL45_L_519.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/damoyolo_tinynasL45_L_519.onnx)|--|

<details>

<summary>Legacy models</summary>

|Model |size |mAP<sup>val<br>0.5:0.95 | Latency T4<br>TRT-FP16-BS1| FLOPs<br>(G)| Params<br>(M)| AliYun Download | Google Download|
| ------        |:---: | :---:     |:---:|:---: | :---: | :---:| :---:|
|[DAMO-YOLO-T](./configs/damoyolo_tinynasL20_T.py) | 640 | 41.8  | 2.78  | 18.1  | 8.5  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL20_T_418.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL20_T_418.onnx)|[torch](https://drive.google.com/file/d/1-9NzCRKJZs3ea_n35seEYSpq3M_RkhcT/view?usp=sharing),[onnx](https://drive.google.com/file/d/1-7s8fqK5KC8z4sXCuh3N900chMtMSYri/view?usp=sharing)|
|[DAMO-YOLO-T*](./configs/damoyolo_tinynasL20_T.py) | 640 | 43.0  | 2.78  | 18.1  | 8.5  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL20_T.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL20_T.onnx) |[torch](https://drive.google.com/file/d/1-6fBf_oe9vITSTYgQkaYklL94REz2zCh/view?usp=sharing),[onnx](https://drive.google.com/file/d/1-1lK83OwVKL4lgHTlbgEZ8pYMYZHhEtE/view?usp=sharing)|
|[DAMO-YOLO-S](./configs/damoyolo_tinynasL25_S.py) | 640 | 45.6  | 3.83  | 37.8  | 16.3  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL25_S_456.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL25_S_456.onnx)|[torch](https://drive.google.com/file/d/1-0GV1lxUS6bLHTOs7aNojsItgjDT6rK8/view?usp=sharing),[onnx](https://drive.google.com/file/d/1--CaKMHm-SjLnprZDMksO-jnbGbV9Zhp/view?usp=sharing)|
|[DAMO-YOLO-S*](./configs/damoyolo_tinynasL25_S.py) | 640 | 46.8  | 3.83  | 37.8  | 16.3 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL25_S.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL25_S.onnx) |[torch](https://drive.google.com/file/d/1-O-ObHN970GRVKkL1TiAxfoMCpYGJS6B/view?usp=sharing),[onnx](https://drive.google.com/file/d/1-NDqCpz2rs1IiKNyIzo1KSxoJACKV65N/view?usp=sharing)|
|[DAMO-YOLO-M](./configs/damoyolo_tinynasL35_M.py) | 640 | 48.7  | 5.62  | 61.8  | 28.2  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL35_M_487.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL35_M_487.onnx)|[torch](https://drive.google.com/file/d/1-RMevyb9nwpDBeTPttiV_iwfsiW_M9ST/view?usp=sharing),[onnx](https://drive.google.com/file/d/1-Cs4ozjAhTH_W32tGnq_L5TfE22vAD_c/view?usp=sharing)|
|[DAMO-YOLO-M*](./configs/damoyolo_tinynasL35_M.py) | 640 | 50.0  | 5.62  | 61.8  | 28.2 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL35_M.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL35_M.onnx)|[torch](https://drive.google.com/file/d/1-RoKaO7U9U1UrweJb7c4Hs_S_qKFDExc/view?usp=sharing),[onnx](https://drive.google.com/file/d/1-HRkLfGoFBjdQDiWudsS1zxicx53Pu5m/view?usp=sharing)|

</details>

- We report the mAP of models on COCO2017 validation set, with multi-class NMS.
- The latency in this table is measured without post-processing(NMS).
- \* denotes the model trained with distillation.
- We use S as teacher to distill T, and M as teacher to distill S, ans L as teacher to distill M, while L is distilled by it self.


### Light Models
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency(ms) CPU<br> OpenVino-Intel8163| FLOPs<br>(G)| Params<br>(M)| AliYun Download | Google Download|
| ------        |:---: | :---:     |:---:|:---: | :---: | :---:| :---:|
| [DAMO-YOLO-Ns](./configs/damoyolo_tinynasL18_Ns.py)| 416| 32.3 | 4.08| 1.56 | 1.41 | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/ckpt/before_distill/damoyolo_nano_small.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/before_distill/damoyolo_tinynasL18_Ns.onnx) | -- |
| [DAMO-YOLO-Nm](./configs/damoyolo_tinynasL18_Nm.py)| 416| 38.2 | 5.05| 3.69 | 2.71 | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/ckpt/before_distill/damoyolo_nano_middle.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/before_distill/damoyolo_tinynasL18_Nm.onnx) | -- |
| [DAMO-YOLO-Nl](./configs/damoyolo_tinynasL20_Nl.py)| 416| 40.5 | 6.69| 6.04 | 5.69 | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/ckpt/before_distill/damoyolo_nano_large.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/onnx/before_distill/damoyolo_tinynasL20_Nl.onnx) | -- |

- We report the mAP of models on COCO2017 validation set, with multi-class NMS.
- The latency in this table is measured without post-processing, following [picodet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6/configs/picodet).
- The latency is evaluated based on [OpenVINO-2022.3.0](https://github.com/openvinotoolkit/openvino), using commands below:
     ```shell
    # onnx export, enable --benchmark to ignore postprocess
    python tools/converter.py -f configs/damoyolo_tinynasL18_Ns.py -c ../damoyolo_tinynasL18_Ns.pth --batch_size 1  --img_size 416 --benchmark
    # model transform
    mo --input_model damoyolo_tinynasL18_Ns.onnx --data_type FP16
    # latency benchmark
    ./benchmark_app -m damoyolo_tinynasL18_Ns.xml -i ./assets/dog.jpg -api sync -d CPU -b 1 -hint latency 
    ```
                                                        

### 701 categories DAMO-YOLO Model
We provide DAMO-YOLO-S model with 701 categories for general object detection, which has been trained on a large dataset including COCO, Objects365 and OpenImage. This model can also serve as a pre-trained model for fine-tuning in downstream tasks, enabling you to achieve better performance with ease.

|Pretrained Model | Downstream Task |mAP<sup>val<br>0.5:0.95 | AliYun Download | Google Download|
| ------        |:---: | :---:     |:---:|:---: |
| 80-categories-DAMO-YOLO-S | VisDrone | 24.6 | [torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL25_S_456.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL25_S_456.onnx) | - |
| 701-categories-DAMO-YOLO-S | VisDrone | 26.6 | [torch](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/Large-Scale-Training/damo_yolo_s_700%2B.pth),[onnx](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/Large-Scale-Training/damo_yolo_s_700%2B.onnx) | - |
- Note: The downloadable model is a pretrained model with 701 categories datasets. We demonstrate the VisDrone results to show that our pretrained model can enhance the performance of downstream tasks.
<div align="left"><img src="assets/701class_cmp_horizontal.png" width="750"></div>

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
pip install cython;
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI # for Windows
```
</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained torch, onnx or tensorRT engine from the benchmark table, e.g., damoyolo_tinynasL25_S.pth, damoyolo_tinynasL25_S.onnx, damoyolo_tinynasL25_S.trt.

Step2. Use -f(config filename) to specify your detector's config, --path to specify input data path, image/video/camera are supported. For example:
```shell
# torch engine with image
python tools/demo.py image -f ./configs/damoyolo_tinynasL25_S.py --engine ./damoyolo_tinynasL25_S.pth --conf 0.6 --infer_size 640 640 --device cuda --path ./assets/dog.jpg

# onnx engine with video
python tools/demo.py video -f ./configs/damoyolo_tinynasL25_S.py --engine ./damoyolo_tinynasL25_S.onnx --conf 0.6 --infer_size 640 640 --device cuda --path your_video.mp4

# tensorRT engine with camera
python tools/demo.py camera -f ./configs/damoyolo_tinynasL25_S.py --engine ./damoyolo_tinynasL25_S.trt --conf 0.6 --infer_size 640 640 --device cuda --camid 0
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

Please refer to [custom dataset tutorial](./assets/CustomDatasetTutorial.md) for details.

</details>



<details>
<summary>Evaluation</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth
```
</details>


<details>
<summary>Customize tinynas backbone</summary>
Step1. If you want to customize your own backbone, please refer to [MAE-NAS Tutorial for DAMO-YOLO](https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/scripts/damo-yolo/Tutorial_NAS_for_DAMO-YOLO_cn.md). This is a detailed tutorial about how to obtain an optimal backbone under the budget of latency/flops.  

Step2. After the searching process completed, you can replace the structure text in configs with it. Finally, you can get your own custom ResNet-like or CSPNet-like backbone after setting the backbone name to TinyNAS_res or TinyNAS_csp. Please notice the difference of out_indices between TinyNAS_res and TinyNAS_csp. 
```
structure = self.read_structure('tinynas_customize.txt')
TinyNAS = { 'name'='TinyNAS_res', # ResNet-like Tinynas backbone
            'out_indices': (2,4,5)}
TinyNAS = { 'name'='TinyNAS_csp', # CSPNet-like Tinynas backbone
            'out_indices': (2,3,4)}

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

Now we support trt_int8 quantization, you can specify trt_type as int8 to export the int8 tensorRT engine. You can also try partial quantization to achieve a good compromise between accuracy and latency. Refer to [partial_quantization](./tools/partial_quantization/README.md) for more details.

Step.1 convert torch model to onnx or trt engine, and the output file would be generated in ./deploy. end2end means to export trt with nms. trt_eval means to evaluate the exported trt engine on coco_val dataset after the export compelete.
```shell
# onnx export 
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640

# trt export
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640 --trt --end2end --trt_eval
```

Step.2 trt engine evaluation on coco_val dataset. end2end means to using trt_with_nms to evaluation.
```shell
python tools/trt_eval.py -f configs/damoyolo_tinynasL25_S.py -trt deploy/damoyolo_tinynasL25_S_end2end_fp16_bs1.trt --batch_size 1 --img_size 640 --end2end
```

Step.3 onnx or trt engine inference demo and appoint test image/video by --path. end2end means to using trt_with_nms to inference.
```shell
# onnx inference
python tools/demo.py image -f ./configs/damoyolo_tinynasL25_S.py --engine ./damoyolo_tinynasL25_S.onnx --conf 0.6 --infer_size 640 640 --device cuda --path ./assets/dog.jpg

# trt inference
python tools/demo.py image -f ./configs/damoyolo_tinynasL25_S.py --engine ./deploy/damoyolo_tinynasL25_S_end2end_fp16_bs1.trt --conf 0.6 --infer_size 640 640 --device cuda --path ./assets/dog.jpg --end2end
```
</details>

## Industry Application Models:
We provide DAMO-YOLO models for applications in real scenarios, which are listed as follows. More powerful models are coming, please stay tuned.

|[**Human Detection**](https://www.modelscope.cn/models/damo/cv_tinynas_human-detection_damoyolo/summary)| [**Helmet Detection**](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_safety-helmet/summary)|[**Head Detection**](https://modelscope.cn/models/damo/cv_tinynas_head-detection_damoyolo/summary) | [**Smartphone Detectioin**](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_phone/summary)|
| :---: | :---: |  :---: | :---: | 
|<img src='./assets/applications/human_detection.png' height="100px" >| <img src='./assets/applications/helmet_detection.png' height="100px">|<img src='./assets/applications/head_detection.png' height="100px"> | <img src='./assets/applications/smartphone_detection.png' height="100px">|
|[**Facemask Detection**](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_facemask/summary) |[**Cigarette Detection**](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_cigarette/summary) |[**Traffic Sign Detection**](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo_traffic_sign/summary) | [**NFL-helmet detection**](https://www.modelscope.cn/models/XianzheXu/cv_tinynas_object-detection_damoyolo_nfl-helmet/summary)|
|<img src='./assets/applications/facemask_detection.png' height="100px">| <img src='./assets/applications/cigarette_detection.png' height="100px">|<img src='./assets/applications/trafficsign_detection.png' height="100px"> | <img src='./assets/applications/nflhelmet_detection.jpg' height="100px">|



## Third Party Resources
In order to promote communication among DAMO-YOLO users, we collect third-party resources in this section. If you have original content about DAMO-YOLO, please feel free to contact us at xianzhe.xxz@alibaba-inc.com.

- DAMO-YOLO Overview: **slides**([中文](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/slides/DAMO-YOLO-Overview.pdf) | [English](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/slides/DAMO-YOLO-Overview-English.pdf)), **videos**([中文](https://www.bilibili.com/video/BV1hW4y1g7za/?spm_id_from=333.337.search-card.all.click) | [English](https://youtu.be/XYQPI7pvMiQ)).
- [DAMO-YOLO Code Interpretation](https://blog.csdn.net/jyyqqq/article/details/128419143)
- [Practical Example for Finetuning on Custom Dataset](https://blog.csdn.net/Cwhgn/article/details/128447380?spm=1001.2014.3001.5501)


## Cite DAMO-YOLO
If you use DAMO-YOLO in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{damoyolo,
   title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
   author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
   journal={arXiv preprint arXiv:2211.15444v2},
   year={2022},
 }

 @inproceedings{sun2022mae,
   title={Mae-det: Revisiting maximum entropy principle in zero-shot nas for efficient object detection},
   author={Sun, Zhenhong and Lin, Ming and Sun, Xiuyu and Tan, Zhiyu and Li, Hao and Jin, Rong},
   booktitle={International Conference on Machine Learning},
   pages={20810--20826},
   year={2022},
   organization={PMLR}
 }

@inproceedings{jiang2022giraffedet,
  title={GiraffeDet: A Heavy-Neck Paradigm for Object Detection},
  author={yiqi jiang and Zhiyu Tan and Junyan Wang and Xiuyu Sun and Ming Lin and Hao Li},
  booktitle={International Conference on Learning Representations},
  year={2022},
}
```

