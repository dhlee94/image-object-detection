import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from random  import random
import os
from torchsummary import summary as summary_
from timm.models.layers import to_2tuple
from einops import rearrange
from einops.layers.torch import Rearrange

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int=0, act_layer=nn.SiLU):
        super(SqueezeExcitation, self).__init__()
        hidden_channels = hidden_channels if hidden_channels else in_channels//16
        self.act_layer = act_layer(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.scale_activation = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)
        y = self.fc1(y)
        y = self.act_layer(y)
        y =self.scale_activation(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class StemBlock(nn.Module):
    def __init__(self , in_channels: int=3, hidden_channels: int=0, out_channels: int=64, act_layer=nn.GELU(), norm_trigger: bool=False, norm=nn.BatchNorm2d, downsample=False, drop_rate=None):
        super(StemBlock, self).__init__()
        if hidden_channels==0:
            hidden_channels = out_channels
        self.act_layer = act_layer
        self.trigger = norm_trigger
        if self.trigger:
            self.norm = norm(hidden_channels)
        stride = 2 if downsample else 1
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.block2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.block1(x)
        if self.trigger:
            x = self.norm(x)
            x = self.act_layer(x)
        x = self.block2(x)
        return x

class MBBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_layer=nn.GELU(), norm=nn.BatchNorm2d, downsample=False, drop_rate=None):
        super(MBBlock, self).__init__()
        self.downsample = downsample
        self.drop_rate = drop_rate
        if downsample:
            self.downlayer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block = nn.Sequential(
            self._make_layer(in_channels=in_channels, out_channels=in_channels*4, act_layer=act_layer, norm=norm, de_trigger=False),
            self._make_layer(in_channels=in_channels*4, out_channels=in_channels*4, act_layer=act_layer, norm=norm, de_trigger=True),
            SqueezeExcitation(in_channels=in_channels*4),
            self._make_layer(in_channels=in_channels*4, out_channels=out_channels, act_layer=None, norm=norm, de_trigger=False)
        )
        self.shortcut = nn.Sequential(
            self._make_layer(in_channels=in_channels, out_channels=out_channels, act_layer=None, norm=norm, de_trigger=False)
        )
        if drop_rate:
            self.droplayer = nn.Dropout(drop_rate)
    def forward(self, x):
        if self.drop_rate:
            x = self.droplayer(x)
        x = self.downlayer(x) if self.downsample else x
        return self.block(x) + self.shortcut(x)
    
    def _make_layer(self, in_channels, out_channels, act_layer, norm, de_trigger):
        layer = nn.ModuleList([])
        layer = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False) if de_trigger
            else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
        if norm:
            layer.append(norm(out_channels))
        if act_layer:
            layer.append(act_layer)
        return nn.Sequential(*layer)
                
class EfficientNet(nn.Module):
    def __init__(self, in_channels: int, out_class: int, img_size: int=224, act_layer=nn.GELU(), norm=nn.BatchNorm2d, 
                 dim=[32, 64, 96, 192, 384, 768], block_size=[2, 2, 3, 5, 2], dropout=0.2):
        super(EfficientNet, self).__init__()
        self.block_size = block_size
        self.features = nn.ModuleList([])
        drop_rate = torch.linspace(0., dropout, sum(block_size)).tolist()
        for idx in range(len(dim)):
            if idx==0:
                self.features.append(self._make_Convlayer(block=StemBlock, in_channels=in_channels, out_channels=dim[idx], act_layer=act_layer, norm=norm, repeat=1))
            else:
                self.features.append(self._make_Convlayer(block=MBBlock, in_channels=dim[idx-1], out_channels=dim[idx], 
                                                          repeat=block_size[idx-1], act_layer=act_layer, norm=norm, drop_rate=drop_rate[sum(block_size[:idx-1]):sum(block_size[:idx])]))
        self.features.append(nn.AdaptiveAvgPool2d(output_size=1))
        self.features.append(nn.Linear(dim[-1], out_class, bias=False))
    def forward(self, x):
        B, _, _, _ = x.size()
        x = self.features[0](x)
        for layer in self.features[1:-2]:
            x = layer(x)
        x = self.features[-2](x)
        x = x.view(B, -1)
        x = self.features[-1](x)
        return x

    def _make_Convlayer(self, block, in_channels, out_channels, repeat, act_layer=nn.SiLU, norm=None, drop_rate=None):
        layers = nn.ModuleList([])
        for num in range(repeat):
            if num==0:
                if drop_rate:
                    layers.append(block(in_channels=in_channels, out_channels=out_channels, downsample=True, act_layer=act_layer, norm=norm, drop_rate=drop_rate[num]))
                else:
                    layers.append(block(in_channels=in_channels, out_channels=out_channels, act_layer=act_layer, norm=norm, downsample=True))
            else:
                layers.append(block(in_channels=out_channels, out_channels=out_channels, drop_rate=drop_rate[num], act_layer=act_layer, norm=norm))
        return nn.Sequential(*layers)