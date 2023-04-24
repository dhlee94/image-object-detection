import os
import yaml
from easydict import EasyDict
import random
import cv2
import numpy as np
import torch
import time
import logging
from pathlib import Path
from torch.nn import functional as F
import torch
import shutil
from collections import OrderedDict
from sklearn import metrics

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))

def DiceAccuracy(score, target, n_classes=4, sigmoid=False, softmax=False, thresh=0.5):
    if softmax:
        score = torch.softmax(score, dim=1).detach()
    elif sigmoid:
        score = torch.sigmoid(score).detach()
    smooth = 1e-5
    score[score>thresh] = 1
    score[score<thresh] = 0
    if n_classes!=1:
        target_list = []
        for i in range(n_classes):
            temp_prob = target == i  # * torch.ones_like(input_tensor)
            target_list.append(temp_prob)
        target = torch.stack(target_list, dim=1)
    score = flatten(score)
    target = flatten(target)
    intersect = torch.sum(score * target.float(), dim=-1)
    y_sum = torch.sum(target, dim=-1)
    z_sum = torch.sum(score, dim=-1)
    tmp = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return tmp.mean()

def IoUAccuracy(score, target, eps=1e-6):
    _, D, _ = score.shape
    iou_acc = 0
    for idx in range(D):
        box1_area = torch.abs((score[:, idx, 2]-score[:, idx, 0]) * (score[:, idx, 3] - score[:, idx, 1]))
        box2_area = torch.abs((target[:, idx, 2]-target[:, idx, 0]) * (target[:, idx, 3] - target[:, idx, 1]))
        
        inter_min_x = torch.max(score[:, idx, 0],target[:, idx, 0])
        inter_min_y = torch.max(score[:, idx, 1],target[:, idx, 1])
        inter_max_x = torch.min(score[:, idx, 2],target[:, idx, 2])
        inter_max_y = torch.min(score[:, idx, 3],target[:, idx, 3])    
        
        inter = torch.clamp((inter_max_x - inter_min_x), min=0) * torch.clamp((inter_max_y - inter_min_y), min=0)
        union = box1_area + box2_area - inter
        iou_acc += (inter / (union+eps)).mean()
    return iou_acc / D

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def make_anchors(Pyramid_length, img_size=256, ratios=None, scales=None):
    #reference to a-PyTorch-Tutorial-to-Object-Detection/utils.py
    Pyramid_size = [img_size//2**(2+i) for i in range(len(Pyramid_length))]
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2**0, 2**(1.0/3.0), 2**(2.0/3.0)])
    num_anchors = len(ratios)*len(scales)
    all_anchors = []
    
    for size in Pyramid_size:        
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = size * np.tile(scales, (2, len(ratios))).T
        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        
        shift_x = (np.arange(0, size) + 0.5) * 1
        shift_y = (np.arange(0, size) + 0.5) * 1

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        tmp_anchors = np.clip((anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))), 0, size)        
        all_anchors.extend(tmp_anchors.reshape((K * A, 4)))
    return all_anchors

def IoUFindBBox(bbox, target, eps=1e-6):
    #reference to a-PyTorch-Tutorial-to-Object-Detection/utils.py
    lower_bounds = torch.max(bbox[:, :2].unsqueeze(1), target[:, :2].unsqueeze(0))
    upper_bounds = torch.max(bbox[:, 2:].unsqueeze(1), target[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds-lower_bounds, min=0)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
    
    area_bbox = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    
    union = area_bbox.unsqueeze(1) + area_target.unsqueeze(0) - intersection
    return intersection / (union+eps)
    
def Calculate_bbox_target_classes(bbox, target, classes, eps=1e-6):
    n_objects = target.size[0]
    overlap = IoUFindBBox(bbox=bbox, target=target, eps=eps)#(n_objects, n_bboxes) IoU
    # For each prior, find the object that has the maximum overlap
    overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (n_bboxes)

    # We don't want a situation where an object is not represented in our positive (non-background) priors -
    # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
    # 2. All priors with the object may be assigned as background based on the threshold (0.5).

    # To remedy this -
    # First, find the prior that has the maximum overlap for each object.
    _, prior_for_each_object = overlap.max(dim=1)  # (n_objects)

    # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
    object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

    # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
    overlap_for_each_prior[prior_for_each_object] = 1.

    # Labels for each prior
    label_for_each_prior = classes[object_for_each_prior]  # (8732)
    # Set priors whose overlaps with objects are less than the threshold to be background (no object)
    label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

    # Store
    true_classes = label_for_each_prior

    # Encode center-size object coordinates into the form we regressed predicted boxes to
    true_locs = cxcy_to_gcxgcy(xy_to_cxcy(boxes[object_for_each_prior]), self.priors_cxcy)  # (8732, 4)
    return true_classes, true_locs

    
