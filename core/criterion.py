import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
from utils.utils import flatten, Calculate_bbox_target_classes
import torchvision
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(target.size(0), -1)
            target = target.view(-1).contiguous()            
        #target = target.view(-1,1)
        #logpt = F.log_softmax(input, dim=-1)
        logpt = F.logsigmoid(input)
        target = target.type(torch.long)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
  
class BoxLoss(nn.Module):#reference to a-PyTorch-Tutorial-to-Object-Detection/utils.py
    def __init__(self, prior_bboxes, bbox_loss=nn.L1Loss(), cross_entropy=nn.CrossEntropyLoss(), threshold=0.5, eps=1e-6):
        super(BoxLoss, self).__init__()
        self.prior_bboxes = prior_bboxes
        self.threshold = threshold
        self.l1Loss = bbox_loss
        self.cross_entropy = cross_entropy
        self.eps = eps

    def _ioUloss(self, target, bbox):
        tmp_target = torch.cat([target[:, :2]-target[:, 2:]/2, target[:, :2]+target[:, 2:]/2], dim=1)
        tmp_bbox = torch.cat([bbox[:, :2]-bbox[:, 2:]/2, bbox[:, :2]+bbox[:, 2:]/2], dim=1)

        lower_bounds = torch.max(tmp_target[:, :2], tmp_bbox[:, :2])
        upper_bounds = torch.min(tmp_target[:, 2:], tmp_bbox[:, 2:])
        intersection_dims = torch.clamp(upper_bounds-lower_bounds, min=0)
        intersection = intersection_dims[:, 0] * intersection_dims[:, 1]
        
        area_bbox = (tmp_bbox[:, 2] - tmp_bbox[:, 0]) * (tmp_bbox[:, 3] - tmp_bbox[:, 1])
        area_target = (tmp_target[:, 2] - tmp_target[:, 0]) * (tmp_target[:, 3] - tmp_target[:, 1])
        
        union = area_target + area_bbox - intersection
        return (1-intersection / (union+self.eps)).mean()
    
    def forward(self, predictlocs, predictcls, target_bboxes, target_labels):
        device = torch.device(f'cuda:{predictlocs.get_device()}')if predictlocs.get_device()>=0 else torch.device('cpu')
        batch_size = predictlocs.size(0)
        n_priors = self.prior_bboxes.size(0)
        n_classes = predictcls.size(2)
        ##target bboxes shape (x_min, y_min, w, h)
        target_bboxes = torch.cat([(target_bboxes[:, :, :2]), (target_bboxes[:, :, :2]+target_bboxes[:, :, 2:])], dim=-1) ##target bboxes shape (x_min, y_min, x_max, y_max)
        ##prior bboxes shape (x_min, y_min, x_max, y_max)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long)
        for idx in range(batch_size):
            tmp_classes, tmp_loc = Calculate_bbox_target_classes(self.prior_bboxes, target_bboxes[idx, ...], target_labels[idx, ...], eps=1e-6, device=device)
            true_locs[idx, ...] = tmp_loc
            true_classes[idx, ...] = tmp_classes
        true_locs = true_locs.to(device)
        true_classes = true_classes.to(device)        
        positive_priors = true_classes != 0
        iou_loss = self._ioUloss(predictlocs[positive_priors], true_locs[positive_priors])
        loc_loss = self.l1Loss(predictlocs[positive_priors], true_locs[positive_priors])
        cross_loss = self.cross_entropy(predictcls.view(-1, n_classes), true_classes.view(-1))
        #return loc_loss, cross_loss, predictlocs[positive_priors], true_locs[positive_priors]
        return loc_loss, cross_loss, iou_loss

