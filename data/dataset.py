import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.utils import parse_data_cfg

class YOLOv2Dataset(Dataset):
    def __init__(self, cfg, training):
        self.data_cfg = parse_data_cfg(cfg.DATA_CFG)
        if training:
            path = self.data_cfg["train"]
        else:
            path = self.data_cfg["valid"]
        with open(path, "r") as file:
            self.img_files = file.readlines()
        if self.data_cfg['names'].find('voc') != -1:
            self.label_files = [
                path.replace("JPEGImages", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]
        else:  # suppose it's COCO
            self.label_files = [
                path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]

        self.batch_size = cfg.BATCH_SIZE
        self.n_cpu = cfg.N_CPU
        self.batch_count = 0
        self.img_size = 416
        self.training = training
        if training:
            self.multiscale = cfg.multiscale_training
            if self.multiscale:
                self.multiscale_interval = 10
                self.min_scale = 10 * 32
                self.max_scale = 19 * 32
        self.jitter, self.saturation, self.exposure, self.hue = self.parse_augmentation_cfg(cfg.MODEL_CFG)

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')
        boxes = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        else:
            print(label_path)
            targets = torch.zeros((0, 6))
        if self.training:
            img, boxes = data_augmentation(img, boxes, self.jitter, self.hue, self.saturation, self.exposure)

        return img, targets, img_path

    def collate_fn(self, batch):
        imgs, targets, img_paths = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.training and self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_scale, self.max_scale + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([transforms.ToTensor()(img.resize((self.img_size, self.img_size))) for img in imgs])
        self.batch_count += 1

        return imgs, targets, img_paths

    def __len__(self):
        return len(self.img_files)

    def parse_augmentation_cfg(self, model_cfg_path):
        """
        Parses the darknet layer configuration file and returns values for data augmentation
        """
        model_cfg_file = open(model_cfg_path, 'r')
        lines = model_cfg_file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]  # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        for line in lines:
            if not line.startswith('['):  # This marks the start of a new block
                key, value = line.split("=")
                key = key.strip()
                if key == 'jitter':
                    jitter = value.strip()
                elif key == 'saturation':
                    saturation = value.strip()
                elif key == 'exposure':
                    exposure = value.strip()
                elif key == 'hue':
                    hue = value.strip()

        return jitter, saturation, exposure, hue

    # def resize(self, img_path):
    #     img = Image.open(img_path).convert('RGB')
    #     img = img.resize((self.img_size, self.img_size))
    #
    #     return img


def data_augmentation(img, boxes, jitter, hue, saturation, exposure):
    # --- crop img according to jitter ---
    origin_height = img.height
    origin_width = img.width

    width_jitter_range = int(origin_width * jitter)
    height_jitter_range = int(origin_height * jitter)

    left_crop_pixel = random.randint(-width_jitter_range, width_jitter_range)
    right_crop_pixel = random.randint(-width_jitter_range, width_jitter_range)
    top_crop_pixel = random.randint(-height_jitter_range, height_jitter_range)
    bottom_crop_pixel = random.randint(-height_jitter_range, height_jitter_range)

    cropped_width = origin_width - left_crop_pixel - right_crop_pixel
    cropped_height = origin_height - top_crop_pixel - bottom_crop_pixel

    img = img.crop((left_crop_pixel, top_crop_pixel, left_crop_pixel + cropped_width - 1, top_crop_pixel + cropped_height - 1))

    # ---- adjust label boxes ----
    # get xyxy pixel coord
    x1_pixel = (boxes[:, 1] - boxes[:, 3] / 2) * origin_width
    y1_pixel = (boxes[:, 2] - boxes[:, 4] / 2) * origin_height
    x2_pixel = (boxes[:, 1] + boxes[:, 3] / 2) * origin_width
    y2_pixel = (boxes[:, 2] + boxes[:, 4] / 2) * origin_height
    # adjust them according to cropped pixel
    x1_pixel -= left_crop_pixel
    y1_pixel -= top_crop_pixel
    x2_pixel -= left_crop_pixel
    y2_pixel -= top_crop_pixel
    # constrain them inside the img
    x1_pixel[x1_pixel < 0], y1_pixel[y1_pixel < 0], x2_pixel[x2_pixel < 0], y2_pixel[y2_pixel < 0] = 0, 0, 0, 0
    x1_pixel[x1_pixel > cropped_width], y1_pixel[y1_pixel > cropped_height], x2_pixel[x2_pixel > cropped_width], y2_pixel[y2_pixel > cropped_height] = cropped_width, cropped_height, cropped_width, cropped_height
    # return to xywh pixel coord
    x_pixel = (x1_pixel + x2_pixel) / 2
    y_pixel = (y1_pixel + y2_pixel) / 2
    w_pixel = x2_pixel - x1_pixel
    h_pixel = y2_pixel - y1_pixel
    # boxes saves the targets' ratio of whole img
    boxes[:, 1] = x_pixel / cropped_width
    boxes[:, 2] = y_pixel / cropped_height
    boxes[:, 3] = w_pixel / cropped_width
    boxes[:, 4] = h_pixel / cropped_height
    # drop bad target
    boxes = boxes[boxes[:, 3] > 0.001]
    boxes = boxes[boxes[:, 4] > 0.001]
    # randomly filp img
    flip = random.randint(1, 10000) % 2
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 1] = 0.999 - boxes[:, 1]
    # adjust in HSV color space
    img = random_distort_image(img, hue, saturation, exposure)

    return img, boxes


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    return im


def rand_scale(s):
    """convert from darknet"""
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2):
        return scale
    return 1./scale


def get_imgs_size(imgs_path):
    sizes = []
    for img_path in imgs_path:
        img = Image.open(img_path).convert('RGB')
        h, w = img.height, img.width
        sizes.append((w, h, w, h))
    return torch.FloatTensor(sizes)