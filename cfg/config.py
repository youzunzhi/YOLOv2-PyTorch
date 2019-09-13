from yacs.config import CfgNode as CN

_C = CN()
# path to data cfg file
_C.DATA_CFG = "cfg/coco.data"
# path to model cfg file
_C.MODEL_CFG = "cfg/yolov2.cfg"
# path to network weights file
_C.WEIGHTS = "weights/yolov2.weights"
# image size to test
_C.IMG_SIZE = 416
# size of each image batch
_C.BATCH_SIZE = 32
# the threshold of non-max suppression algorithm
_C.NMS_THRESH = 0.4
# only keep detections with conf higher than conf_thresh
_C.CONF_THRESH = 0.5
# total train epochs
_C.TOTAL_EPOCHS = 160
# Folder to save checkpoints and log
_C.OUTPUT_DIR = "runs/"
# use CUDA or not
_C.USE_CUDA = False
# GPU device ID
_C.GPU = '2,3'
# number of cpu threads to use during batch generation
_C.N_CPU = 8
# interval of evaluations on validation set
_C.EVAL_INTERVAL = 1
# interval of saving model weights
_C.SAVE_INTERVAL = 1
# use remote debugger to debug
_C.DEBUG = False

cfg = _C


