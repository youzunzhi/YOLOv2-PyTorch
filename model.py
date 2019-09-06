import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import os, time, logging

from modules import *
from utils.computation import *
from utils.utils import parse_data_cfg, draw_detect_box, log_train_progress, show_eval_result
from data.dataset import get_imgs_size

class YOLOv2Model(object):
    def __init__(self, cfg, training=False):
        self.cfg = cfg
        self.layer_defs = self.parse_model_cfg(cfg.MODEL_CFG)
        self.data_cfg = parse_data_cfg(cfg.DATA_CFG)
        self.hyper_parameters, self.network = self.get_network()
        self.batch_size = cfg.BATCH_SIZE
        self.training = training
        if training:
            self.seen = 0
            self.header_info = np.array([0, 0, 0, self.seen], dtype=np.int32)

            self.learning_rate = float(self.hyper_parameters['learning_rate'])

            decay = float(self.hyper_parameters['decay'])
            self.optimizer = optim.SGD(self.modules.parameters(),
                                       lr=self.learning_rate/self.batch_size,
                                       momentum=float(self.hyper_parameters['momentum']),
                                       weight_decay=decay*self.batch_size)

        pretrained_weights = cfg.WEIGHTS
        self.load_weights(pretrained_weights)

    def __call__(self, imgs, *args, **kwargs):
        x = imgs
        layer_outputs = []
        for i, (layer_def, layer) in enumerate(zip(self.layer_defs, self.network)):
            if layer_def['type'] in ['convolutional', 'maxpool', 'reorg']:
                x = layer(x)
            elif layer_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in layer_def["layers"].split(",")], 1)
            elif layer_def['type'] == 'region':
                output = layer(x)

            layer_outputs.append(x)

        return output

    def detect(self, img_path):
        self.set_eval_state()
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.cfg.IMG_SIZE, self.cfg.IMG_SIZE))
        img = transforms.ToTensor()(img)
        img = torch.stack([img])
        img_size = get_imgs_size([img_path])
        img_size = torch.stack([img_size])

        output = self(img)
        predictions = non_max_suppression(output, img_size, self.cfg.CONF_THRESH, self.cfg.NMS_THRESH)
        draw_detect_box(img_path, predictions[0], self.data_cfg['names'])

    def eval(self, dataloader):
        self.set_eval_state()
        metrics = []
        labels = []
        for batch_i, (imgs, targets, imgs_path) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            labels += targets[:, 1].tolist()
            if self.cfg.USE_CUDA:
                imgs = imgs.cuda()
            imgs_size = get_imgs_size(imgs_path)
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            for target in targets:
                target[2:] *= imgs_size[target[0].long()]
            with torch.no_grad():
                outputs = self(imgs)
                outputs = outputs.cpu()

            predictions = non_max_suppression(outputs, imgs_size, self.cfg.CONF_THRESH, self.cfg.NMS_THRESH)
            metrics += get_batch_metrics(predictions, targets)

        show_eval_result(metrics, labels)


    def train(self, train_dataloader, eval_dataloader):
        total_epochs = self.options.total_epochs
        self.set_train_state()
        for epoch in range(total_epochs):
            start_time = time.time()
            for batch_i, (imgs, targets, img_path) in enumerate(train_dataloader):
                if self.options.use_cuda:
                    inputs = imgs.cuda()
                    targets = targets.cuda()
                for module in self.modules[:-1]:
                    inputs = module(inputs)
                loss = self.modules[-1](inputs, self.seen, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log_train_progress(epoch, total_epochs, batch_i, len(train_dataloader), self.learning_rate, start_time,
                                   self.modules[-1].metrics)

            if epoch % self.options.eval_interval == self.options.eval_interval - 1:
                self.eval(eval_dataloader)
            if epoch % self.options.save_interval == self.options.save_interval - 1:
                fname = os.path.join(self.options.save_path, self.save_weights_fname)
                self.save_weights(fname)
            self.adjust_learning_rate(epoch)

        if total_epochs % self.options.eval_interval != self.options.eval_interval - 1:
            self.logger.print_log("\n---- Evaluating Model ----")
            self.eval(eval_dataloader)
        if total_epochs % self.options.save_interval != self.options.save_interval - 1:
            self.logger.print_log("\n---- Saving Model ----")
            fname = os.path.join(self.options.save_path, self.save_weights_fname)
            self.save_weights(fname)


    def parse_model_cfg(self, model_cfg_path):
        """
        Parses the yolov2-tiny layer configuration file and returns module definitions(list of dicts)
        """
        model_cfg_file = open(model_cfg_path, 'r')
        lines = model_cfg_file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]  # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        layer_defs = []
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                layer_defs.append({})
                layer_defs[-1]['type'] = line[1:-1].rstrip()
                if layer_defs[-1]['type'] == 'convolutional':
                    layer_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                layer_defs[-1][key.rstrip()] = value.strip()

        return layer_defs

    def get_network(self):
        """
            Constructs network(nn.ModuleList) of layer blocks from module configuration in layer_defs
        """
        hyper_parameters = self.layer_defs.pop(0)
        output_filters = [int(hyper_parameters["channels"])]
        network = nn.ModuleList()
        for layer_i, layer_def in enumerate(self.layer_defs):
            if layer_def["type"] == "convolutional":
                layer = nn.Sequential()
                bn = int(layer_def["batch_normalize"])
                filters = int(layer_def["filters"])
                kernel_size = int(layer_def["size"])
                stride = int(layer_def["stride"])
                pad = (kernel_size - 1) // 2
                layer.add_module(
                    f"conv_{layer_i}",
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad,
                        bias=not bn,
                    ),
                )
                if bn:
                    layer.add_module(
                        f"batchnorm_{layer_i}",
                        nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
                    )
                if layer_def["activation"] == "leaky":
                    layer.add_module(
                        f"leaky_{layer_i}",
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                elif layer_def["activation"] != "linear":
                    assert False, "unknown activation type"
            elif layer_def["type"] == "maxpool":
                pool_size = int(layer_def['size'])
                stride = int(layer_def['stride'])
                if stride > 1:
                    layer = nn.MaxPool2d(pool_size, stride)
                else:
                    layer = MaxPoolStride1()
            elif layer_def["type"] == "reorg":
                stride = int(layer_def['stride'])
                filters = stride * stride * output_filters[-1]
                layer = ReorgLayer(stride)
            elif layer_def["type"] == "route":
                layer_idx = [int(x) for x in layer_def["layers"].split(",")]
                filters = sum([output_filters[1:][i] for i in layer_idx])
                layer = EmptyLayer()
            elif layer_def["type"] == "region":
                layer = RegionLayer(layer_def)
            else:
                assert False, 'unknown type %s' % (layer_def['type'])

            if self.cfg.USE_CUDA:
                layer = layer.cuda()
            network.append(layer)
            output_filters.append(filters)

        return hyper_parameters, network

    def adjust_learning_rate(self, epoch):
        if epoch == 60:
            self.learning_rate *= 0.1
        elif epoch == 90:
            self.learning_rate *= 0.1
        else:
            return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate / self.batch_size

    def load_weights(self, weights_file):
        """Parses and loads the weights stored in 'weights_file'"""
        assert os.path.exists(weights_file), "weights_file not exists"
        # Open the weights file
        with open(weights_file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=4)  # First four are header values
            self.header_info = header  # Needed to write header when saving weights
            if self.training:
                self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet19_448.conv.23" in weights_file:
            cutoff = 24

        ptr = 0
        for i, (layer_def, layer) in enumerate(zip(self.layer_defs, self.network)):
            if i == cutoff:
                break
            if layer_def['type'] == 'convolutional':
                conv_layer = layer[0]
                bn = int(layer_def["batch_normalize"])
                if bn:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = layer[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, fname, cutoff=-1):
        """
        :param path: path of the new weights file
        :param cutoff: save layers between 0 and cutoff (cutoff == -1 -> all save)
        :return:
        """
        fp = open(fname, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, module in enumerate(self.modules[:cutoff]):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                if not isinstance(self.modules[i + 1], nn.BatchNorm2d):
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                    conv_layer.weight.data.cpu().numpy().tofile(fp)

            elif isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
        logger = logging.getLogger('YOLOv2.Train')
        self.logger.print_log('Saved weights to '+fname)

    def set_train_state(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def set_eval_state(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()
