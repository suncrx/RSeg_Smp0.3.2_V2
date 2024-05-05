# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:04:21 2024

@author: renxi
"""

import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    def __init__(self,in_channels, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channels,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,
                      padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x
    


if __name__ == '__main__':
    eca64 = ECA(64)
    #eca128 = ECA(128)
    print(eca64)
    #print(eca128)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', 
                           pretrained=True)
    layer1 = model.layer1 
    print(model.layer1)
    
    #add se layers to layer1
    model.layer1 = nn.Sequential(layer1[0], eca64, 
                                 layer1[1], eca64)
    print(model.layer1)
    
    model.fc = nn.Linear(512, 6)
    print(model)
    
    x = torch.rand(4, 3,128,128)
    o = model(x)    