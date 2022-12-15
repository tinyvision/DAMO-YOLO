# Tutorial for finetuning on a custom dataset

## Step 1. Convert the custom dataset to coco format.

- The coco format can be found in [Link](https://cocodataset.org/#format-data). Specifically, the coco annotation file (`.json`) includes three necessary feilds, *i.e.*, image, annotation, categories. A toy sample (named `toy_sample.json`) is provided below:

```json
{
  "categories": 
  [{
      "supercategory": "person", 
      "id": 1, 
      "name": "person"
  }], 
 "images": 
  [{
      "license": 1, 
      "file_name": "000000425226.jpg",        
      "coco_url": "http://images.cocodataset.org/val2017/000000425226.jpg", 
      "height": 640, 
      "width": 480, 
      "date_captured": 
      "2013-11-14 21:48:51", 
      "flickr_url": 
      "http://farm5.staticflickr.com/4055/4546463824_bc40e0752b_z.jpg", 
      "id": 1
  }], 
 "annotations": 
  [{
      "image_id": 1, 
      "category_id": 1, 
      "segmentation": [], 
      "area": 47803.279549999985, 
      "iscrowd": 0, 
      "bbox": [73.35, 206.02, 300.58, 372.5], 
      "id": 1
  }]
}
```

- Then we can organize the custom dataset (including images and annotations) as follows:
```
├── Custom_coco
│   ├── annotations
│   │   └── toy_sample.json
│   ├── images
│   │   └── 000000425226.jpg
```

## Step2. Link custom dataset into DAMO-YOLO
- Link your dataset into `datasets`. 
```
ln -s path/to/Custom_coco datasets/toy_sample
```

- Add the custom dataset into `damo/config/paths_catalog.py`. Note, the added dataset should contain **coco** in their names to declare the dataset format, *e.g.*, here we use `sample_train_coco` and `sample_test_coco`.
```
'sample_train_coco': {
    'img_dir': 'toy_sample/images',
    'ann_file': 'toy_sample/annotations/toy_sample.json'
},
'sample_test_coco': {
    'img_dir': 'toy_sample/images',
    'ann_file': 'toy_sample/annotations/toy_sample.json'
},
```


## Step3. Modify the config file.
In this tutorial, we finetune on DAMO-YOLO-Tiny as example.
- Download the DAMO-YOLO-Tiny torch model from [Model Zoo](https://github.com/tinyvision/DAMO-YOLO#Model-Zoo)
- Add the following pretrained model path into `damoyolo_tinynasL20_T.py`.
```
self.train.finetune_path='path/to/damoyolo_tinynasL20_T.pth'
```
- Modify the custom dataset in config file. Change `coco_2017_train` and `coco_2017_test` in `damoyolo_tinynasL20_T.py` to `sample_train_coco` and `sample_test_coco` respectively.
https://github.com/tinyvision/DAMO-YOLO/blob/6e38813220900955d0f6138429c91a33a79c922f/configs/damoyolo_tinynasL20_T.py#L33-L34

- Modify the category number in config file. Change `'num_classes': 80` in `damoyolo_tinynasL20_T.py` to `'num_classes': 1`. Because in our toy sample, there is only one category, so we set `num_classes` to 1.
https://github.com/tinyvision/DAMO-YOLO/blob/6e38813220900955d0f6138429c91a33a79c922f/configs/damoyolo_tinynasL20_T.py#L64-L66

## Step4. Finetune on custom dataset
You can run the finetuning with the following code:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL20_T.py
```

