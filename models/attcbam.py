# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:22:58 2024

@author: renxi
"""

"""
convolutional block attention module (CBAM)
Original paper addresshttps: https://arxiv.org/pdf/1807.06521.pdf
Time: 2024-02-28
"""
import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)
        

if __name__ == '__main__':
    cbam64 = CBAM(64)
    cbam128 = CBAM(128)
    cbam256 = CBAM(256)
    cbam512 = CBAM(512)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 
                           'resnet18', 
                           pretrained=True)
    print(model.layer1)
    
    model.layer1 = nn.Sequential(model.layer1[0], cbam64, 
                                 model.layer1[1], cbam64)
    model.layer2 = nn.Sequential(model.layer2[0], cbam128, 
                                 model.layer2[1], cbam128)
    model.layer3 = nn.Sequential(model.layer3[0], cbam256,
                                 model.layer3[1], cbam256)
    model.layer4 = nn.Sequential(model.layer4[0], cbam512, 
                                 model.layer4[1], cbam512)
    
    model.fc = nn.Linear(512, 6)
    print(model)
    
    x = torch.rand(4, 3,128,128)
    o = model(x)  
    
    