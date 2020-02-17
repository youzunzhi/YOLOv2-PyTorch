from yacs.config import CfgNode as CN
import torch
_C = CN()
_C.OUTPUT_DIR = 'runs/'
_C.EXPERIMENT_NAME = 'default'
_C.CUDA_ID = '0'
train_on = 'coco'
if train_on == 'coco':
    _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2.cfg"
    _C.WEIGHTS_FNAME = "weights/yolov2.weights"
elif train_on == 'voc':
    _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2-voc.cfg"
    _C.WEIGHTS_FNAME = "weights/yolov2-voc.weights"
_C.CONF_THRESH = 0.2
_C.NMS_THRESH = 0.4
_C.EVAL_INTERNAL = 100
_C.SAVE_INTERNAL = 50

_C.DATA = CN()
if train_on == 'coco':
    if torch.cuda.is_available():
        _C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco.data"
    else:
        _C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco-mac.data"
elif train_on == 'voc':
    _C.DATA.DATA_CFG_FNAME = "pjreddie_files/voc.data"
_C.DATA.BATCH_SIZE = 64
_C.DATA.MULTISCALE = True
_C.DATA.N_CPU = 0

_C.TRAIN = CN()
_C.TRAIN.TOTAL_EPOCHS = 400
