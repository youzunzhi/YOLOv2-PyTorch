import argparse, time, os
from cfg.config import cfg
from utils.utils import setup_logger
from model import YOLOv2Model
from data import make_dataloader

import warnings
def main():
    warnings.simplefilter("always")
    parser = argparse.ArgumentParser(description="YOLOv2 Train")
    parser.add_argument("--config_file", default="cfg/train-voc.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config cfg using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.USE_CUDA:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

    time_string = '[{}]'.format(time.strftime('%Y-%m-%d-%X', time.localtime(time.time())))
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, time_string)
    cfg.freeze()

    logger = setup_logger("YOLOv2", cfg.OUTPUT_DIR, 0)
    logger.info("Running with config:\n{}".format(cfg))

    train_dataloader, eval_dataloader = make_dataloader(cfg, training=True)
    model = YOLOv2Model(cfg, training=True)

    model.train(train_dataloader, eval_dataloader)




if __name__ == '__main__':
    main()