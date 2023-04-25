import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from timm.models.layers import to_2tuple
import json

class ImageDataset(Dataset):
    def __init__(self, Image_path, num_class=1, one_hot=True, transform=None, class_names=['물달개비', '한련초', '흰명아주']):
        self.Image_path = Image_path
        self.transform = transform
        self.num_class = num_class
        self.one_hot = one_hot
        self.class_names = class_names
    def __len__(self):
        return len(self.Image_path['image'])
    
    def __getitem__(self, idx):
        data = Image.open(self.Image_path['image'][idx]).convert('RGB')
        with open(self.Image_path['label'][idx], 'r', encoding="UTF-8") as f:
            label = json.load(f)
        cls = [self.class_names.index(label['info']['label_path'].split('/')[1])+1]
        w, h = np.array(data).shape[:2]
        bbox = np.array(label['annotations']['bbox'])
        if len(bbox.shape)==1:
            bbox = [label['annotations']['bbox']] # x, y, w, h
            bbox[0][2] = w-1 if bbox[0][2]>=w else bbox[0][2]
            bbox[0][3] = h-1 if bbox[0][3]>=h else bbox[0][3]
        else:
            bbox = label['annotations']['bbox'].tolist() # x, y, w, h
            for idx in range(len(bbox)):
                bbox[idx][2] = w-1 if bbox[idx][2]>=w else bbox[idx][2]
                bbox[idx][3] = h-1 if bbox[idx][3]>=h else bbox[idx][3]
        if self.transform != None:
            transformed = self.transform(image=np.array(data), bboxes=bbox, class_labels=cls)
            return transformed['image'], torch.FloatTensor(transformed['bboxes']), torch.LongTensor(transformed['class_labels'])
        else:
            return torch.FloatTensor(data.permute(2, 0, 1)), torch.FloatTensor(bbox).unsqueeze(0), torch.LongTensor(cls)

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    img, bbox, clas = zip(*data)
    lengths = [len(cap) for cap in bbox]
    
    img = torch.stack(img, 0)
    targets_bbox = torch.zeros(len(bbox), max(lengths), 4).long()
    for idx, cap in enumerate(bbox):
        end = lengths[idx]
        targets_bbox[idx, :end, ...] = cap[:end, ...]
        
    targets_cls = torch.zeros(len(clas), max(lengths)).long()
    for idx, cap in enumerate(clas):
        end = lengths[idx]
        targets_cls[idx, :end] = cap[:end]
    
    return img, targets_bbox, targets_cls