from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = 'runs/'
_C.EXPERIMENT_NAME = 'default'
_C.CUDA_ID = '0'
_C.MODEL_CFG_FNAME = "pjreddie_files/yolov2.cfg"
_C.WEIGHTS_FNAME = "weights/yolov2.weights"
_C.CONF_THRESH = 0.5
_C.NMS_THRESH = 0.4

_C.DATA = CN()
_C.DATA.DATA_CFG_FNAME = "pjreddie_files/coco.data"
_C.DATA.BATCH_SIZE = 32
_C.DATA.MULTISCALE = True
_C.DATA.N_CPU = 8
