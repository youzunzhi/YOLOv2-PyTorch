from yacs.config import CfgNode as CN
import torch

_C = CN()
_C.OUTPUT_DIR = 'runs/'
_C.EXPERIMENT_NAME = 'coco_pretrain'
_C.CUDA_ID = '0'
_C.EVAL_INTERVAL = 10
_C.SAVE_INTERVAL = 'best'

train_on = 'coco'
if train_on == 'coco':
    _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2.cfg"
elif train_on == 'voc':
    # _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2-voc.cfg"
    _C.MODEL_CFG_FNAME = "pjreddie_files/yolov2-tiny-voc.cfg"
_C.WEIGHTS_FNAME = "weights/darknet19_448.conv.23"
# _C.WEIGHTS_FNAME = "weights/yolov2-tiny-voc.weights"
# _C.WEIGHTS_FNAME = "weights/yolov2.weights"
# _C.WEIGHTS_FNAME = "no"

# for eval
_C.CONF_THRESH = 0.005
_C.NMS_THRESH = 0.45

_C.DATA = CN()
if train_on == 'coco':
    if torch.cuda.is_available():
        _C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco.data"
    else:
        _C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco-mac.data"
elif train_on == 'voc':
    _C.DATA.DATA_CFG_FNAME = "pjreddie_files/voc.data"
_C.DATA.IMG_SIZE = 416
_C.DATA.BATCH_SIZE = 4
_C.DATA.MULTISCALE = False
_C.DATA.N_CPU = 0

_C.TRAIN = CN()
_C.TRAIN.TOTAL_EPOCHS = 400
_C.TRAIN.LEARNING_RATE = 0.0001
_C.TRAIN.LR_STEP = 0.1
_C.TRAIN.LR_STEP_EPOCH = [320, 350]
_C.TRAIN.DONTCARE = True # don't care about the nasty situation

