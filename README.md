# YOLOv2-PyTorch
Here' s another implementation of YOLOv2 in PyTorch. I intend to implement a highly readable codebase of YOLOv2 to
reveal the essence of this model. So, if you find anything confusing please feel free to open an issue to discuss with me.

## Getting Started
### Installation
```shell script
git clone https://github.com/youzunzhi/YOLOv2-PyTorch.git 
```
### Prerequisites
> python 3.x \
> PyTorch >= 1.0.1 \
> [yacs](https://github.com/rbgirshick/yacs)

### Download Pretrained weights
```shell script
cd weights/ 
wget -c https://pjreddie.com/media/files/yolov2.weights # yolov2 trained on coco
wget -c https://pjreddie.com/media/files/yolov2-tiny-voc.weights # yolov2-tiny trained on voc
wget -c https://pjreddie.com/media/files/darknet19_448.conv.23 # darknet backbone pretrained on ImageNet
```
### Get Datasets
#### Pascal VOC
##### Download the dataset
```shell script
cd <path-to-voc>/
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
```shell script
wget https://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Modify Cfg for Pascal VOC
Change the cfg/voc.data config file
```
classes= 20
train  = <path-to-voc>/train.txt
valid  = <path-to-voc>/2007_test.txt
names = data/voc.names
backup = backup
```

#### COCO

##### Download the dataset

```shell script
cd <path-to-coco>/
# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# Download Images
wget -c https://pjreddie.com/media/files/train2014.zip
wget -c https://pjreddie.com/media/files/val2014.zip

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip

cd ..

# Download COCO Metadata
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
```

##### Modify Cfg for COCO

Change the cfg/coco.data config file

```
classes= 80
train  = <path-to-coco>/trainvalno5k.txt
valid  = <path-to-coco>/5k.txt
names = data/coco.names
backup = backup
```


## Evaluation

You can modify the configs in config/eval_cfg.py or in command line following the usage of yacs. For example:

Evaluate YOLOv2 on COCO
```shell script
python eval.py MODEL_CFG_FNAME pjreddie_files/yolov2-voc.cfg \
WEIGHTS_FNAME weights/yolov2-voc.weights \
DATA.DATA_CFG_FNAME pjreddie_files/voc.data
```

Evaluate  YOLOv2-tiny on Pascal VOC with image size of 416x416
```shell script
python eval.py MODEL_CFG_FNAME pjreddie_files/yolov2-tiny-voc.cfg \
WEIGHTS_FNAME weights/yolov2-tiny-voc.weights \
DATA.DATA_CFG_FNAME pjreddie_files/voc.data \
DATA.IMG_SIZE 416
```

| Model       | Dataset | Image Size | mAP (this implementation) | mAP (paper) |
| ----------- | ------- | ---------- | ------------------------- | ----------- |
| YOLOv2      | COCO    | 608        | 46.9                      | 48.1        |
| YOLOv2-tiny | VOC     | 416        | 57.3                      | 57.1        |

(Got `nan` when evaluating YOLOv2 on VOC. I guess there's something wrong with yolov2-voc.weights on [pjreddie.com](https://pjreddie.com/media/files/yolov2-voc.weights).)



## Training

You can modify the configs in config/train_cfg.py or in command line following the usage of yacs. For example:

Train on Pascal VOC with pretrained darknet backbone
```shell script
python train.py MODEL_CFG_FNAME pjreddie_files/yolov2-voc.cfg \
WEIGHTS_FNAME weights/darknet19_448.conv.23 \
DATA.DATA_CFG_FNAME pjreddie_files/voc.data
```

Train on COCO from scratch with Multi-Scale Training technique and 
don't care about the nasty situation (see [model/modules.py, line 134](https://github.com/youzunzhi/YOLOv2-PyTorch/blob/98352ff18c8a9bcde4e2d07505fd30da589a4abc/model/modules.py#L134))

```shell script
python train.py MODEL_CFG_FNAME pjreddie_files/yolov2.cfg \
WEIGHTS_FNAME no \
DATA.DATA_CFG_FNAME pjreddie_files/coco.data\
DATA.MULTISCALE True
TRAIN.DONTCARE True
```



## Credits

I referred to many fantastic repos and blogs during the implementation:

[eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

[marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)

[pjreddie/darknet](https://github.com/pjreddie/darknet)

[Training Object Detection (YOLOv2) from scratch using Cyclic Learning Rates](https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755)

[目标检测|YOLOv2原理与实现(附YOLOv3)](https://zhuanlan.zhihu.com/p/35325884)