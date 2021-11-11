from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def prediction_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = prediction.size(2)
    bb_attrs = 5 + num_classes
    num_anchors = len(anchors)
    prediction = prediction.reshape(batch_size, grid_size * grid_size, bb_attrs * num_anchors)
    prediction = prediction.view(batch_size, -1, bb_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # t_x
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    # t_y
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    # P_c
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    grid = np.arange(grid_size)
    x_co, y_co = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x_co).view(-1,1)
    y_offset = torch.FloatTensor(y_co).view(-1,1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    # height, width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # sigmoid class score
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # from grid size to input size
    prediction[:,:,:4] *= stride

    return prediction
    
def write_result(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = False
    for index in range(batch_size):
        pred = prediction[index]
        max_conf, max_conf_index = torch.max(pred[:, 5:], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        pred = torch.cat((pred[:, :5], max_conf, max_conf_index), 1)
        non_zero_ind = torch.nonzero(pred[:, 4])
        try:
            pred_ = pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue
        pred_classes = unique(pred_[:, -1])

        for cls in pred_classes:
            cls_mask = pred_ * (pred_[:, -1] == cls).float().unsqueeze(1)
            cls_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            pred_cls = pred_[cls_mask_ind].view(-1, 7)

            conf_sort_ind = torch.sort(pred_cls[:, -2], descending=True)[1]
            pred_cls = pred_cls[conf_sort_ind]
            num_detec = pred_cls.size(0)

            for i in range(num_detec):
                try:
                    ious = bbox_iou(pred_cls[i].unsqueeze(0), pred_cls[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                pred_cls[i+1:] *= iou_mask

                non_zero_ind = torch.nonzero(pred_cls[:, 4]).squeeze()
                pred_cls = pred_cls[non_zero_ind].view(-1, 7)
            
            batch_ind = pred_cls.new(pred_cls.size(0), 1).fill_(index)

            out = torch.cat((batch_ind, pred_cls), 1)
            if not write:
                output = out
                write = True
            else:
                output = torch.cat((output, out))
    try:
        return output
    except:
        return 0
        

def unique(x):
    tensor_np = x.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = x.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box, other):
    b1_x1, b1_y1, b1_x2, b1_y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = other[:, 0], other[:, 1], other[:, 2], other[:, 3]

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    b1_area = (b1_x1 - b1_x2) * (b1_y1 - b1_y2)
    b2_area = (b2_x1 - b2_x2) * (b2_y1 - b2_y2)

    iou = intersection / (b1_area + b2_area - intersection)

    return iou


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

