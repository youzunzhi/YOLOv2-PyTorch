from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = 'runs/'
_C.EXPERIMENT_NAME = 'default'
_C.CUDA_ID = '0'
_C.MODEL_CFG_FNAME = "pjreddie_files/yolov2-voc.cfg"
_C.WEIGHTS_FNAME = "weights/yolov2-voc.weights"
_C.CONF_THRESH = 0.5
_C.NMS_THRESH = 0.4

_C.DATA = CN()
_C.DATA.DATA_CFG_FNAME = "pjreddie_files/voc.data"
_C.DATA.BATCH_SIZE = 1
_C.DATA.MULTISCALE = False
_C.DATA.N_CPU = 0
