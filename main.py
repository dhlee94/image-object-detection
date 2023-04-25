from utils.utils import seed_everything, make_anchors
import argparse
from core.criterion import BoxLoss
from core.function import train_epoch
import pickle
import numpy as np
from core.optimizer import CosineAnnealingWarmUpRestarts
from data.dataset import ImageDataset
from timm.models.layers import to_2tuple
from torch.utils.data import DataLoader
from models.EfficientDet import EfficientDet
import torch
import torch.nn as nn
import os
import albumentations
import albumentations.pytorch as transforms
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import pandas as pd
import cv2

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int,
                    default=304, help='random seed')
parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='Train and Valid Csv file directory path')
parser.add_argument('--log_path', type=str, default='./log', help='Save log file path')
parser.add_argument('--img_size', default=128, type=int, help='Resize img size')
parser.add_argument('--gpu', default="0", type=str, help='use gpu num')
parser.add_argument('--workers', default=0, type=int, help='dataloader workers')
parser.add_argument('--epoch', default=100, type=int, help='Train Epoch')
parser.add_argument('--batch_size', default=4, type=int, help='Train Batch size')
parser.add_argument('--in_channels', default=3, type=int, help='input image channels')
parser.add_argument('--out_channels', default=3, type=int, help='Number of class')
parser.add_argument('--filter_size', default=[32, 64, 128, 256, 512], help='ResUnet Filter size')
parser.add_argument('--optim', default="SGD", type=str, help='type of optimizer')
parser.add_argument('--momentum', default=0.95, type=float, help='SGD momentum')
parser.add_argument('--lr', default=1e-3, type=float, help='Train Learning Rate')
parser.add_argument('--optimizer_eps', default=1e-8, type=float, help='AdamW optimizer eps')
parser.add_argument('--optimizer_betas', default=(0.9, 0.999), help='AdamW optimizer betas')
parser.add_argument('--weight_decay', default=0.95, type=float, help='Optimizer weight decay parameter')
parser.add_argument('--scheduler', default="LambdaLR", type=str, help='type of Scheduler')
parser.add_argument('--lambda_weight', default=0.975, type=float, help='LambdaLR Scheduler lambda weight')
parser.add_argument('--t_scheduler', default=80, type=int, help='CosineAnnealingWarmUpRestarts optimizer time step')
parser.add_argument('--trigger_scheduler', default=1, type=int, help='CosineAnnealingWarmUpRestarts optimizer T trigger')
parser.add_argument('--eta_scheduler', default=1.25e-3, type=float, help='CosineAnnealingWarmUpRestarts optimizer eta max')
parser.add_argument('--up_scheduler', default=8, type=int, help='CosineAnnealingWarmUpRestarts optimizer time Up')
parser.add_argument('--gamma_scheduler', default=0.5, type=float, help='CosineAnnealingWarmUpRestarts optimizer gamma')
parser.add_argument('--model_path', default=None, type=str, help='retrain model load path')
parser.add_argument('--ratios', default=None, help='Anchors Ratios')
parser.add_argument('--scales', default=None, help='Anchors scales')
parser.add_argument('--model_save_path', default='./weight', type=str, help='model save path')
parser.add_argument('--write_iter_num', default=20, type=int, help='write learning situation iteration time')

def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Reading Data")
    train_csv = pd.read_csv(os.path.join(args.csv_path, 'Train_data.csv'))
    valid_csv = pd.read_csv(os.path.join(args.csv_path, 'Valid_data.csv'))
    
    image_shape = to_2tuple(args.img_size)
    height, width = image_shape
    train_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=0.3),
                    albumentations.ShiftScaleRotate(p=0.3, rotate_limit=90),
                    albumentations.VerticalFlip(p=0.3),
                    albumentations.RandomBrightnessContrast(p=0.3),
                    albumentations.GaussNoise(p=0.3)                    
                ],p=1),
                albumentations.Normalize(),
                transforms.ToTensorV2()
            ], bbox_params=albumentations.BboxParams(format='coco', label_fields=['class_labels'])
        )
    valid_transform = albumentations.Compose(
            [
                albumentations.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=1),
                    albumentations.VerticalFlip(p=1)
                ],p=1),
                albumentations.Normalize(),
                transforms.ToTensorV2()
            ], bbox_params=albumentations.BboxParams(format='coco', label_fields=['class_labels'])
        )
        
    #del data_file
    train_dataset = ImageDataset(Image_path=train_csv, transform=train_transform)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=True)
    valid_dataset = ImageDataset(Image_path=valid_csv, transform=valid_transform)
    validloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True, shuffle=False)
    
    #model init
    print("Model Init")
    model = EfficientDet(in_channels=args.in_channels, out_class=args.out_channels+1, img_size=args.img_size, feature_size=64, act_layer=nn.GELU(), norm=nn.BatchNorm2d, 
                         dim=[32, 64, 96, 192, 384, 768], block_size=[2, 2, 3, 5, 2], dropout=0.2, epsilon=1e-4, num_anchors=9)
    
    if args.optim=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.AdamW(model.parameters(), eps=args.optimizer_eps, betas=args.optimizer_betas,
                                        lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler=='LambdaLR':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:args.lambda_weight**epoch)
    else:
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.t_scheduler, T_mult=args.trigger_scheduler, 
                                                  eta_max=args.eta_scheduler, T_up=args.up_scheduler, gamma=args.gamma_scheduler)
    
    prior_bboxes = torch.FloatTensor(make_anchors(img_size=args.img_size, ratios=args.ratios, scales=args.scales, Pyramid_length=[2, 2, 3, 5, 2])).to(device)
    criterion = BoxLoss(prior_bboxes=prior_bboxes, threshold=0.5)
    
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location={'cuda:0':'cpu'})
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best_loss = 0
    
    train_epoch(model=model, write_iter_num=5, trainloader=trainloader, validloader=validloader, optimizer=optimizer, scheduler=scheduler, device=device, 
                criterion=criterion, start_epoch=start_epoch, end_epoch=args.epoch, log_path=args.log_path, model_path=args.model_save_path, best_loss=best_loss)
    

if __name__ == '__main__':
    main()