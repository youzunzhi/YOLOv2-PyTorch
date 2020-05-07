# YOLOv2-PyTorch
Here' s another implementation of YOLOv2 in PyTorch. I intend to implement a highly readable codebase of YOLOv2 to
reveal the essence of this model. So, if you find anything confusing please feel free to open an issue to discuss with me.

## Getting Started
### Installation
```shell script
git clone https://github.com/youzunzhi/YOLOv2-PyTorch.git 
```
### Prerequisites
> python 3.x\
> PyTorch >= 1.0.1\
> [yacs](https://github.com/rbgirshick/yacs)

### Download Pretrained weights
```shell script
cd weights/ 
wget -c https://pjreddie.com/media/files/yolov2-voc.weights # weights trained on voc
wget -c https://pjreddie.com/media/files/darknet19_448.conv.23 # weights pretrained on ImageNet
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
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```shell script
classes= 20
train  = <path-to-voc>/train.txt
valid  = <path-to-voc>/2007_test.txt
names = data/voc.names
backup = backup
```

## Evaluation
