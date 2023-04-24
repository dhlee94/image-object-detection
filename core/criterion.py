import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function, Variable
from utils.utils import flatten, Calculate_bbox_target_classes

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
        
class BoxLoss(nn.Module):
    #reference to a-PyTorch-Tutorial-to-Object-Detection/utils.py
    def __init__(self, prior_bboxes, threshold=0.5):
        super(BoxLoss, self).__init__()
        self.prior_bboxes = prior_bboxes
        self.threshold = threshold
        self.l1Loss = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forawrd(self, prediction_loc, prediction_cls, target_bboxes, target_labels, device):
        batch_size = prediction_loc.size(0)
        n_priors = self.prior_bboxes.size(0)
        n_classes = prediction_cls.size(2)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for idx in range(batch_size):
            tmp_classes, tmp_loc = Calculate_bbox_target_classes(self.prior_bboxes, target_bboxes, target_labels, eps=1e-6)
            true_locs[idx, ...] = tmp_loc
            true_classes[idx, ...] = tmp_classes
        
        positive_priors = true_classes != 0

        loc_loss = self.l1Loss(prediction_loc[positive_priors], true_locs[positive_priors])
        cross_loss = self.cross_entropy(prediction_cls.view(-1, n_classes), true_classes.view(-1))

        return loc_loss + cross_loss