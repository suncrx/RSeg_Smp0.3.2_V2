# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:02:50 2024

@author: renxi
"""
import torch
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1):
    ''' 1x1 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    ''' 3x3 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, 
                     dilation=dilation, bias=False)

def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    ''' 7x7 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                     stride=stride, padding=padding, dilation=dilation, 
                     bias=False)


class BAM(nn.Module):
    def __init__(self, in_channels, reduction=8, 
                 dilation=2):
        super(BAM, self).__init__()
        self.hid_channels = in_channels // reduction
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channels, 
                             out_features=self.hid_channels)
        self.bn1_1d = nn.BatchNorm1d(self.hid_channels)
        self.fc2 = nn.Linear(in_features=self.hid_channels, 
                             out_features=in_channels)
        self.bn2_1d = nn.BatchNorm1d(in_channels)

        self.conv1 = conv1x1(in_channels, self.hid_channels)
        self.bn1_2d = nn.BatchNorm2d(self.hid_channels)
        self.conv2 = conv3x3(self.hid_channels, self.hid_channels, 
                             stride=1, padding=self.dilation, 
                             dilation=self.dilation)
        self.bn2_2d = nn.BatchNorm2d(self.hid_channels)
        self.conv3 = conv3x3(self.hid_channels, self.hid_channels, 
                             stride=1, padding=self.dilation, 
                             dilation=self.dilation)
        self.bn3_2d = nn.BatchNorm2d(self.hid_channels)
        self.conv4 = conv1x1(self.hid_channels, 1)
        self.bn4_2d = nn.BatchNorm2d(1)


    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.bn1_1d(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.bn2_1d(Mc)
        Mc = self.relu(Mc)

        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.bn1_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.bn2_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.bn3_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.bn4_2d(Ms)
        Ms = self.relu(Ms)

        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))
        Mf = 1 + self.sigmoid(Mc * Ms)
        return x * Mf


if __name__ == '__main__':
    
    m = BAM(32)
    #print(un)
    d = torch.rand(4, 32, 256,256)
    print(d.shape)
    o=m(d)
    print(o.shape)