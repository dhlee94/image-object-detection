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

def IoUAccuracy(bbox, target, eps=1e-6):
    B, _, _ = bbox.shape###score (c_x, c_y, w, h)
    iou_acc = 0
    for idx in range(B):
        lower_bounds = torch.max(target[idx, :, :2].unsqueeze(1), bbox[idx, :, :2].unsqueeze(0))
        upper_bounds = torch.max(target[idx, :, 2:].unsqueeze(1), bbox[idx, :, 2:].unsqueeze(0))
        intersection_dims = torch.clamp(upper_bounds-lower_bounds, min=0)
        intersection = intersection_dims[idx, :, :, 0] * intersection_dims[idx, :, :, 1]
        
        area_bbox = (bbox[idx, :, 2] - bbox[idx, :, 0]) * (bbox[idx, :, 3] - bbox[idx, :, 1])
        area_target = (target[idx, :, 2] - target[idx, :, 0]) * (target[idx, :, 3] - target[idx, :, 1])
        
        union = area_target.unsqueeze(1) + area_bbox.unsqueeze(0) - intersection
        iou_acc += intersection / (union+eps)
    return iou_acc/B

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

def IoUFindBBox(bbox, target, eps=1e-6):##shape (x1, y1, x2, y2)
    #reference to a-PyTorch-Tutorial-to-Object-Detection/utils.py
    lower_bounds = torch.max(target[:, :2].unsqueeze(1), bbox[:, :2].unsqueeze(0))
    upper_bounds = torch.min(target[:, 2:].unsqueeze(1), bbox[:, 2:].unsqueeze(0), )
    intersection_dims = torch.clamp(upper_bounds-lower_bounds, min=0)
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
    
    area_bbox = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    
    union = area_target.unsqueeze(1) + area_bbox.unsqueeze(0) - intersection
    return intersection / (union+eps)
    
def Calculate_bbox_target_classes(bbox, target, classes, eps=1e-6, threshold=0.5, device=None):
    n_objects = target.shape[0]
    overlap = IoUFindBBox(bbox=bbox, target=target, eps=eps)#(n_objects, n_bboxes) IoU
    # For each prior, find the object that has the maximum overlap
    overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (n_bboxes)
    overlap_for_each_prior = overlap_for_each_prior.to(device)
    object_for_each_prior = object_for_each_prior.to(device)
    # We don't want a situation where an object is not represented in our positive (non-background) priors -
    # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
    # 2. All priors with the object may be assigned as background based on the threshold (0.5).

    # To remedy this -
    # First, find the prior that has the maximum overlap for each object.
    _, prior_for_each_object = overlap.max(dim=1)  # (n_objects)
    prior_for_each_object = prior_for_each_object.to(device)
    # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
    object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

    # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
    overlap_for_each_prior[prior_for_each_object] = 1.

    # Labels for each prior
    label_for_each_prior = classes[object_for_each_prior]  # (8732)
    # Set priors whose overlaps with objects are less than the threshold to be background (no object)
    label_for_each_prior[overlap_for_each_prior < threshold] = 0  # (8732)

    # Store
    true_classes = label_for_each_prior

    # Encode center-size object coordinates into the form we regressed predicted boxes to
    tmp_target = target[object_for_each_prior]
    tmp_target = torch.cat([(tmp_target[:, :2] + tmp_target[:, 2:])/2, (tmp_target[:, 2:] - tmp_target[:, :2])], dim=1)## (x_min, y_min, x_max, y_max) -> (c_x, x_y, w, h)
    tmp_bbox = torch.cat([(bbox[:, :2] + bbox[:, 2:])/2, (bbox[:, 2:] - bbox[:, :2])], dim=1)## (x_min, y_min, x_max, y_max) -> (c_x, x_y, w, h)
    true_locs = torch.cat([(tmp_target[:, :2]-tmp_bbox[:, :2])/(tmp_bbox[:, 2:]/10), torch.log(tmp_target[:, 2:]/(tmp_bbox[:, 2:]))*5], dim=1)  # (8732, 4)
    true_locs[true_locs.isnan()] = 0
    return true_classes, true_locs

def gcxgcy_to_cxcy(gcxgcy, priors_xy):
    ##reference to github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    priors_cxcy = torch.cat([(priors_xy[:, :2] + priors_xy[:, 2:])/2, (priors_xy[:, 2:] - priors_xy[:, :2])], dim=1)## (x_min, y_min, x_max, y_max) -> (c_x, x_y, w, h)
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:]/10 + priors_cxcy[:, :2], torch.exp(gcxgcy[:, 2:]/5)*priors_cxcy[:, 2:]], dim=1)

def IoUAccuracy(predict_bboxes, bboxes, eps=1e-6):
    pxy_bboxes = torch.zeros(predict_bboxes.shape)
    xy_bboxes = torch.zeros(bboxes.shape)
    for idx in range(len(predict_bboxes)):
        pxy_bboxes[idx, ...] = torch.cat([predict_bboxes[idx, :, :2]-predict_bboxes[idx, :, 2:]/2, predict_bboxes[idx, :, :2]+predict_bboxes[idx, :, 2:]/2], dim=1)
        xy_bboxes[idx, ...] = torch.cat([bboxes[idx, :, :2]-bboxes[idx, :, 2:]/2, bboxes[idx, :, :2]+bboxes[idx, :, 2:]/2], dim=1)
    iou = torch.zeros(predict_bboxes.shape[:2])
    for idx in range(len(predict_bboxes)):
        lower_inter = torch.min(pxy_bboxes[idx, :, 2:], xy_bboxes[idx, :, 2:])
        max_inter = torch.max(pxy_bboxes[idx, :, :2], xy_bboxes[idx, :, :2])
        intersection = (max_inter[:, 0]-lower_inter[:, 0])*(max_inter[:, 1]-lower_inter[:, 1])
        max_union = torch.min(pxy_bboxes[idx, :, :2], xy_bboxes[idx, :, :2])
        lower_union = torch.max(pxy_bboxes[idx, :, 2:], xy_bboxes[idx, :, 2:])
        union = (max_union[:, 0]-lower_union[:, 0])*(max_union[:, 1]-lower_union[:, 1])
        tmp = union/(intersection+1e-6)
        tmp[tmp.isnan()]=0
        iou[idx, :] = tmp
    return iou.mean()