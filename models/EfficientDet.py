import torch
import torch.nn as nn
import torch.nn.functional as F
from random  import random
import os
from models.EfficientNet import EfficientNet
class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.GELU(), norm=nn.BatchNorm2d):
        super(DepthwiseConvBlock,self).__init__()
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.group_layer = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn = norm(out_channels)
        self.act = act_layer
        
    def forward(self, x):
        x = self.layer(x)
        x = self.group_layer(x)
        x = self.bn(x)
        return self.act(x)
        
class BiPFN_block(nn.Module):
    def __init__(self, feature_size=64, act_layer=nn.GELU(), norm=nn.BatchNorm2d, epsilon=1e-4):
        super(BiPFN_block, self).__init__()
        self.epsilon = epsilon

        self.p4p3_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.p3p2_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.p2p1_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.p1p0_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p3p4_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.p2p3_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.p1p2_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.p0p1_layer = DepthwiseConvBlock(in_channels=feature_size, out_channels=feature_size, act_layer=act_layer, norm=norm)
        self.downsample = nn.MaxPool2d(kernel_size=2)
        
        # TODO : Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = act_layer
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = act_layer
    def forward(self, x):
        p4_x, p3_x, p2_x, p1_x, p0_x = x

        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon

        p3_tmp = self.p4p3_layer(w1[0, 0]*p3_x + w1[1, 0]*self.upsample(p4_x))
        p2_tmp = self.p3p2_layer(w1[0, 1]*p2_x + w1[1, 1]*self.upsample(p3_tmp))
        p1_tmp = self.p2p1_layer(w1[0, 2]*p1_x + w1[1, 2]*self.upsample(p2_tmp))
        p0_tmp = self.p1p0_layer(w1[0, 3]*p0_x + w1[1, 3]*self.upsample(p1_tmp))

        p1_out = self.p0p1_layer(w2[0, 0]*p1_x + w2[1, 0]*p1_tmp + w2[2, 0]*self.downsample(p0_tmp))
        p2_out = self.p1p2_layer(w2[0, 1]*p2_x + w2[1, 1]*p2_tmp + w2[2, 1]*self.downsample(p1_tmp))
        p3_out = self.p2p3_layer(w2[0, 2]*p3_x + w2[1, 2]*p3_tmp + w2[1, 2]*self.downsample(p2_tmp))
        p4_out = self.p3p4_layer(w2[0, 3]*p4_x + w2[1, 3]*self.downsample(p3_tmp))

        return [p0_tmp, p1_out, p2_out, p3_out, p4_out]

class BiFPN(nn.Module):
    def __init__(self, in_channels, feature_size=64, act_layer=nn.GELU(), norm=nn.BatchNorm2d, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # TODO : Init weights
        self.p4_layer = nn.Conv2d(in_channels=in_channels[4], out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_layer = nn.Conv2d(in_channels=in_channels[3], out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.p2_layer = nn.Conv2d(in_channels=in_channels[2], out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.p1_layer = nn.Conv2d(in_channels=in_channels[1], out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.p0_layer = nn.Conv2d(in_channels=in_channels[0], out_channels=feature_size, kernel_size=1, stride=1, padding=0)
        self.bifpn_layer = BiPFN_block(feature_size=feature_size, act_layer=act_layer, norm=norm, epsilon=epsilon)
    def forward(self, x):
        p0, p1, p2, p3, p4 = x

        p4_x = self.p4_layer(p4)
        p3_x = self.p3_layer(p3)
        p2_x = self.p2_layer(p2)
        p1_x = self.p1_layer(p1)
        p0_x = self.p0_layer(p0)
        features = [p4_x, p3_x, p2_x, p1_x, p0_x]
        
        return self.bifpn_layer(features)

class classification(nn.Module):
    def __init__(self, in_channels, num_classes=1000, num_anchors=9, act_layer=nn.GELU(), norm=nn.BatchNorm2d):
        super(classification, self).__init__()
        self.num_anchors = num_anchors
        self.num_class = num_classes
        self.layer1 = nn.Sequential(
            norm(in_channels),
            act_layer,
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.layer2 = nn.Sequential(
            norm(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes*num_anchors, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 3, 1)
        B, W, H, _ = x.shape
        x = x.view(B, W, H, self.num_anchors, self.num_class)

        return x.contiguous().view(B, -1, self.num_class)

class regression(nn.Module):
    def __init__(self, in_channels, num_anchors=9, act_layer=nn.GELU(), norm=nn.BatchNorm2d):
        super(regression, self).__init__()
        self.num_anchors = num_anchors
        self.layer1 = nn.Sequential(
            norm(in_channels),
            act_layer,
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.layer2 = nn.Sequential(
            norm(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=num_anchors*4, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 3, 1)
        B, W, H, _ = x.shape
        x = x.view(B, W, H, self.num_anchors, 4)

        return x.contiguous().view(B, -1, 4)

class EfficientDet(nn.Module):
    def __init__(self, in_channels: int, out_class: int, img_size: int=256, feature_size=64, act_layer=nn.GELU(), norm=nn.BatchNorm2d, 
                 dim=[32, 64, 96, 192, 384, 768], block_size=[2, 2, 3, 5, 2], dropout=0.2, epsilon=1e-4, num_anchors=9):
        super(EfficientDet, self).__init__()
        self.feature_extractor = EfficientNet(in_channels=in_channels, out_class=out_class, img_size=img_size, act_layer=act_layer, norm=norm,
                                              dim=dim, block_size=block_size, dropout=dropout).features[:-2]
        self.bifpn = BiFPN(in_channels=dim[1:], feature_size=feature_size, act_layer=act_layer, norm=norm, epsilon=epsilon)
        self.classification_model = classification(in_channels=feature_size, num_classes=out_class, num_anchors=num_anchors, act_layer=act_layer, norm=norm)
        self.regression_model = regression(in_channels=feature_size, num_anchors=num_anchors, act_layer=act_layer, norm=norm)
    def forward(self, x):
        B, _, _, _ = x.size()
        x  = self.feature_extractor[0](x)
        P = []
        for layer in self.feature_extractor[1:]:
            x = layer(x)
            P.append(x)
        features = self.bifpn(P)
        classification = torch.cat([self.classification_model(feature) for feature in features], dim=1)
        regression = torch.cat([self.regression_model(feature) for feature in features], dim=1)
        return classification, regression
