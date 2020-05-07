from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import os, time, logging

from model.networks import YOLOv2Network
from utils.computation import *
from utils.utils import parse_data_cfg, draw_detect_box, log_train_progress, show_eval_result_and_get_mAP
torch.manual_seed(0)


class YOLOv2Model(object):
    def __init__(self, cfg, training=False):
        self.cfg = cfg
        self.training = training
        self.use_cuda = torch.cuda.is_available()
        self.network = YOLOv2Network(cfg.MODEL_CFG_FNAME, cfg.WEIGHTS_FNAME, self.use_cuda)
        if training:
            self.save_weights_fname_prefix = os.path.join(self.cfg.OUTPUT_DIR, cfg.EXPERIMENT_NAME)
            self.seen = 0
            self.learning_rate = cfg.TRAIN.LEARNING_RATE
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=self.learning_rate,
                                       momentum=0.9,
                                       weight_decay=0.0005)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.cfg.TRAIN.LR_STEP_EPOCH, self.cfg.TRAIN.LR_STEP)

    def detect(self, img_path):
        self.network.eval()

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.cfg.IMG_SIZE, self.cfg.IMG_SIZE))
        img = transforms.ToTensor()(img)
        img = torch.stack([img])

        with torch.no_grad():
            output = self.network(img)
        predictions = non_max_suppression(output, self.cfg.CONF_THRESH, self.cfg.NMS_THRESH)
        draw_detect_box(img_path, predictions[0], parse_data_cfg(self.cfg.DATA_CFG_FNAME)['names'])

    def eval(self, eval_dataloader):
        self.network.eval()
        metrics = []
        labels = []
        for batch_i, (imgs, targets, imgs_path) in enumerate(tqdm.tqdm(eval_dataloader, desc="Detecting objects")):
            labels += targets[:, 1].tolist()
            if self.use_cuda:
                imgs = imgs.cuda()
            with torch.no_grad():
                outputs = self.network(imgs)
            predictions = non_max_suppression(outputs, self.cfg.CONF_THRESH, self.cfg.NMS_THRESH)
            metrics += get_batch_metrics(predictions, targets)
        mAP = show_eval_result_and_get_mAP(metrics, labels)
        return mAP

    def train(self, train_dataloader, eval_dataloader):
        total_epochs = self.cfg.TRAIN.TOTAL_EPOCHS
        self.network.train()
        best_mAP = 0
        for epoch in range(1, total_epochs+1):
            start_time = time.time()
            for batch_i, (imgs, targets, img_paths) in enumerate(train_dataloader):
                if self.use_cuda:
                    imgs = imgs.cuda()
                    targets = targets.cuda().detach()

                loss = self.network(imgs, targets, img_paths, self.cfg.TRAIN.DONTCARE)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                log_train_progress(epoch, batch_i, len(train_dataloader), self.optimizer.param_groups[0]['lr'], start_time,
                                   self.network.module_list[-1].metrics)
                if loss > 10:
                    logger = logging.getLogger('YOLOv2.Train')
                    logger.info(f'loss got too high. May caused by one of {img_paths}.')
                    return

            self.scheduler.step()
            if not isinstance(self.cfg.SAVE_INTERVAL, str) and (epoch % self.cfg.SAVE_INTERVAL == 0 or epoch == 1):
                epoch_save_weights_fname = f'{self.save_weights_fname_prefix}-{epoch}.weights'
                self.network.save_weights(epoch_save_weights_fname)
                logger = logging.getLogger('YOLOv2.Train')
                logger.info(f'saved weight to {epoch_save_weights_fname}')
            if epoch % self.cfg.EVAL_INTERVAL == 0 or epoch == 1:
                mAP = self.eval(eval_dataloader)
                if self.cfg.SAVE_INTERVAL == 'best':
                    if mAP > best_mAP:
                        best_mAP = mAP
                        self.network.save_weights(f'{self.save_weights_fname_prefix}-best.weights')
                        logger = logging.getLogger('YOLOv2.Train')
                        logger.info(f'saved weight to {self.save_weights_fname_prefix}-best.weights')

        if total_epochs % self.cfg.EVAL_INTERVAL != 0:
            mAP = self.eval(eval_dataloader)
            if mAP > best_mAP:
                self.network.save_weights(f'{self.cfg.EXPERIMENT_NAME}-best.weights')