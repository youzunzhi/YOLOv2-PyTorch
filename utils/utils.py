import argparse
import os, sys, time, datetime, math, shutil
import logging
import numpy as np
import torch, random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from terminaltables import AsciiTable
from utils.computation import ap_per_class


def prepare_experiment(cfg, log_prefix):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    # each experiment's output is in the dir named after the time when it starts to run
    if torch.cuda.is_available():
        log_dir_name = log_prefix + '-[{}]'.format(
            (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%m%d%H%M%S'))
    else:
        log_dir_name = log_prefix + '-[{}]'.format((datetime.datetime.now()).strftime('%m%d%H%M%S'))
    log_dir_name += cfg.EXPERIMENT_NAME
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, log_dir_name)
    os.mkdir(cfg.OUTPUT_DIR)
    cfg.freeze()

    logger = setup_logger("YOLOv2", cfg.OUTPUT_DIR, 0)
    logger.info("Running with config:\n{}".format(cfg))
    return cfg


def handle_keyboard_interruption(cfg):
    assert cfg.OUTPUT_DIR != 'runs/'
    assert cfg.OUTPUT_DIR.find('[') != -1
    save = input('save the log files(%s)?[y|n]' % cfg.OUTPUT_DIR)
    if save == 'y':
        print('log files may be saved in', cfg.OUTPUT_DIR)
    elif save == 'n':
        shutil.rmtree(cfg.OUTPUT_DIR)
        print('log directory removed:', cfg.OUTPUT_DIR)
    else:
        print('unknown input, saved by default')


def handle_other_exception(cfg):
    import traceback
    print(traceback.format_exc())
    assert cfg.OUTPUT_DIR != 'runs/'
    assert cfg.OUTPUT_DIR.find('[') != -1
    print('log directory removed:', cfg.OUTPUT_DIR)
    shutil.rmtree(cfg.OUTPUT_DIR)


def setup_logger(name, log_path, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(message)s", '%m%d%H%M%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_path:
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        # ISOTIMEFORMAT = '%Y-%m-%d-%X'
        # txt_name = '{}.txt'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
        txt_name = 'log.txt'
        fh = logging.FileHandler(os.path.join(log_path, txt_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_class_names(names_file):
    class_names = []
    with open(names_file, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def parse_data_cfg(path):
    """Parses the data configuration file"""
    data_cfg = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        data_cfg[key.strip()] = value.strip()
    return data_cfg


def draw_detect_box(img_path, predictions, names_file, save_path):
    class_names = load_class_names(names_file)

    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    unique_labels = predictions[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)

    img = np.array(Image.open(img_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # rescale pred_boxes to image size
    predictions[:, 0] *= img.shape[1]
    predictions[:, 1] *= img.shape[0]
    predictions[:, 2] *= img.shape[1]
    predictions[:, 3] *= img.shape[0]

    for prediction in predictions:
        center_x, center_y, img_w, img_h, class_id = prediction[0], prediction[1], prediction[2], \
                                                                           prediction[3], prediction[-1]
        left_top_x = center_x - img_w / 2
        left_top_y = center_y - img_h / 2
        patch_color = bbox_colors[int(np.where(unique_labels == int(class_id))[0])]
        bbox_patch = patches.Rectangle((left_top_x, left_top_y), img_w, img_h,
                                 linewidth=2, edgecolor=patch_color, facecolor="none")
        ax.add_patch(bbox_patch)
        plt.text(left_top_x, left_top_y, s=class_names[int(class_id)], color="white", verticalalignment="top",
                    bbox={"color": patch_color, "pad": 0})
    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    save_fname = os.path.join(save_path, img_path.split('/')[-1])
    plt.savefig(save_fname, bbox_inches="tight", pad_inches=0.0)
    print('saved detection to', save_fname)
    plt.close()


# def draw_detect_box_deprecate(img_path, predictions, names_file):
#     def get_color(channel, offset, class_num):
#         colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
#         ratio = float(offset) / class_num * 5
#         i = int(math.floor(ratio))
#         j = int(math.ceil(ratio))
#         ratio = ratio - i
#         r = (1 - ratio) * colors[i][channel] + ratio * colors[j][channel]
#         return int(r * 255)
#
#     class_names = load_class_names(names_file)
#
#     def get_rgb(class_id):
#         class_num = len(class_names)
#         offset = class_id * 123457 % class_num
#         red = get_color(2, offset, class_num)
#         green = get_color(1, offset, class_num)
#         blue = get_color(0, offset, class_num)
#         rgb = (red, green, blue)
#         return rgb
#
#     img = cv2.imread(img_path)
#     save_path = 'detect-' + img_path.split('/')[-1]
#     for prediction in predictions:
#         left_top_x, left_top_y, right_bottom_x, right_bottom_y, class_id = prediction[0], prediction[1], prediction[2], \
#                                                                            prediction[3], prediction[-1]
#         color_tuple = get_rgb(class_id)
#         cv2.rectangle(img, (left_top_x, left_top_y), (right_bottom_x, right_bottom_y), color_tuple, 4)
#         cv2.putText(img, class_names[int(class_id)], (left_top_x, left_top_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     color_tuple)
#     cv2.imwrite(save_path, img)
#     print('save detection to', save_path)


def log_train_progress(epoch, batch_i, total_batch, lr, start_time, metrics):
    log_str = f"[Epoch {epoch}, Batch {batch_i}, LR {lr:.0e}] "
    formats = {m: "%.6f" for m in metrics}
    formats["grid_size"] = "%d"
    formats["cls_acc"] = "%.2f%%"
    for i, metric in enumerate(metrics):
        row_metrics = formats[metric] % metrics.get(metric, 0)
        log_str += f'{metric}:{row_metrics},'
    # Determine approximate time left for epoch
    epoch_batches_left = total_batch - (batch_i + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
    log_str += f"ETA:{time_left}"

    logger = logging.getLogger('YOLOv2.Train')
    logger.info(log_str)


def log_train_progress_ascii_table(epoch, total_epochs, batch_i, total_batch, lr, start_time, metrics):
    log_str = "\n---- [Epoch %d/%d, Batch %d/%d, LR " % (epoch, total_epochs, batch_i, total_batch) + str(
        lr) + "] ----\n"
    metric_table = [["Metrics", "Region Layer"]]
    formats = {m: "%.6f" for m in metrics}
    formats["grid_size"] = "%2d"
    formats["cls_acc"] = "%.2f%%"
    for i, metric in enumerate(metrics):
        row_metrics = formats[metric] % metrics.get(metric, 0)
        metric_table += [[metric, row_metrics]]
    log_str += AsciiTable(metric_table).table

    # Determine approximate time left for epoch
    epoch_batches_left = total_batch - (batch_i + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
    log_str += f"\n---- ETA {time_left}"

    logger = logging.getLogger('YOLOv2.Train')
    logger.info(log_str)


def show_eval_result_and_get_mAP(metrics, labels):
    true_positives, pred_conf, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_conf, pred_labels, labels)
    logger = logging.getLogger('YOLOv2.Eval')
    mAP = AP.mean()
    logger.info(f"mAP: {mAP}")
    return mAP

