import argparse, time, os, torch
from cfg.config import cfg
from model import YOLOv2Model
from data import make_dataloader


def main():
    parser = argparse.ArgumentParser(description="YOLOv2 Detection")
    parser.add_argument("--config_file", default="cfg/eval.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config cfg using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import pydevd_pycharm
    pydevd_pycharm.settrace('140.113.138.32', port=12345, stdoutToServer=True, stderrToServer=True)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    time_string = '[{}]'.format(time.strftime('%Y-%m-%d-%X', time.localtime(time.time())))
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, time_string)
    cfg.freeze()

    eval_dataloader = make_dataloader(cfg, training=False)
    model = YOLOv2Model(cfg)

    model.eval(eval_dataloader)




if __name__ == '__main__':
    main()