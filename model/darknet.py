from __future__ import division

import argparse
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import cv2
import numpy as np
from util import *


def get_test_input(image=None):
    if image.any() == None:
        img = cv2.imread("/content/drive/MyDrive/YOLOv3/data/dog-cycle-car.png")
    else:
        img = image
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                # store the lines in a list
    lines = [x for x in lines if len(x) > 0]       # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']      # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]   # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":           # This marks the start of a new block
            if len(block) != 0:      # If block is not empty, implies it is storing values of previous block.
                blocks.append(block) # add it the blocks list
                block = {}           # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    print(blocks)
    prev_filter = 3
    module_list = nn.ModuleList()
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        x_type = x['type']
        if x_type == 'convolutional':
            active_func = x['activation']
            try:
                use_bn = int(x['batch_normalize'])
                bias = False
            except:
                use_bn = 0
                bias = True
            
            filters = int(x['filters'])
            use_pad = int(x['pad'])
            size = int(x['size'])
            stride = int(x['stride'])

            if use_pad:
                padding = (size - 1) // 2
            else:
                padding = 0

            conv2d = nn.Conv2d(prev_filter, filters, size, stride, padding, bias=bias)
            module.add_module('conv_{}'.format(index), conv2d)

            if use_bn:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(index), bn)
            
            # linear giữ nguyên
            if active_func == 'leaky':
                active = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('Activation_{}'.format(index), active)
        elif x_type == 'upsample':
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('upsample_{}'.format(index), upsample)
        elif x_type == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
        elif x_type == 'route':
            layers = x['layers'].split(',')
            start = int(layers[0])
            try:
                end = int(layers[1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module('route_{}'.format(index), route)

            if end != 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        elif x_type == 'yolo':
            masks = x['mask'].split(',')
            masks = [int(x) for x in masks]

            anchors = x['anchors'].split(', ')
            anchors = [x.split(',') for x in anchors]
            anchors = [anchors[i] for i in masks]
            anchors = [[int(x) for x in tmp] for tmp in anchors]

            yolo = DetectionLayer(anchors)
            module.add_module('yolo_{}'.format(index), yolo)
        
        module_list.append(module)
        prev_filter = filters
        output_filters.append(filters)
    return (net_info, module_list)


class EmptyLayer(pl.LightningModule):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(pl.LightningModule):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(pl.LightningModule):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.lr = 0.001

    def forward(self, x, device):
        x = x.to(device=device)
        modules_info = self.blocks[1:]
        outputs = {}
        write = 0
        for index, module in enumerate(modules_info):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[index](x)
            elif module_type == 'route':
                layers = module['layers'].split(',')
                layers = [int(i) for i in layers]
                start = layers[0]
                try:
                    end = layers[1]
                except:
                    end = 0

                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index

                if end != 0:
                    x1 = outputs[index + start]
                    x2 = outputs[index + end]
                    x = torch.cat((x1, x2), 1)
                else:
                    x = outputs[index + start]   
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[index - 1] + outputs[index + from_]
            elif module_type == 'yolo':
                anchors = self.module_list[index][0].anchors
                input_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.data
                x = prediction_transform(x, input_dim, anchors, num_classes, device)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[index] = x
        return detections
    
    def train_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')

        if fp is None:
            print('Hello')

        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        modules_info = self.blocks[1:]
        for index, module in enumerate(modules_info):
            module_type = module['type']

            if module_type == 'convolutional':
                model = self.module_list[index]
                try:
                    use_bn = int(module['batch_normalize'])
                except:
                    use_bn = 0
                
                conv = model[0]
                if use_bn:
                    bn = model[1]
                    
                    num_bn_biases = bn.bias.numel()

                    # Load weight
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # reshape to correct shape 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr+ num_biases])
                    ptr = ptr + num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]) 
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



























