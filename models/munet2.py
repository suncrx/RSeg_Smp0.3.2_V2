# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:38:24 2022

@author: renxi


A user-defined Unet for segmentation.

!!!!!!!!!!!!!
The output is not processed with activation such as sigmod (for binary segmentatio)
and softmax (for multi-class segmentation). You need to do sigmoiding 
yourself in the case of binary segmentation, and softmaxing multi-class segmentation .

"""


import torch
import torch.nn as nn
#from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kersz=5):
        super(Conv, self).__init__()
        
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, kersz, 1, kersz//2, padding_mode='reflect'),
            nn.BatchNorm2d(C_out),
            #nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv2d(C_out, C_out, kersz, 1, kersz//2, padding_mode='reflect'),
            nn.BatchNorm2d(C_out),
            nn.ReLU(),    
            #nn.Dropout(0.2),            
        )
        

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.MaxPool2d((2,2))

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        #self.Up = nn.Conv2d(C, C // 2, 1, 1)
        self.Up = nn.ConvTranspose2d(C, C//2, 3, 2, padding=1, output_padding=1)

    def forward(self, x, r):
        #up = F.interpolate(x, scale_factor=2, mode="nearest")
        #x = self.Up(up)
        
        x = self.Up(x)

        return torch.cat((x, r), 1)


# when n_classes=1, the model is for binary segmentation
class MUNet2(nn.Module):
    
    def __init__(self, n_classes=1, ker_size=3, activation=None):
        super(MUNet2, self).__init__()

        self.C1 = Conv(3, 16, kersz=ker_size)
        self.D1 = DownSampling(16)
        self.C2 = Conv(16, 32, kersz=ker_size)
        self.D2 = DownSampling(32)
        self.C3 = Conv(32, 64, kersz=ker_size)
        self.D3 = DownSampling(64)
        self.C4 = Conv(64, 128, kersz=ker_size)
        self.D4 = DownSampling(128)
        self.C5 = Conv(128, 256, kersz=ker_size)

        self.U1 = UpSampling(256)
        self.C6 = Conv(256, 128, kersz=ker_size)
        self.U2 = UpSampling(128)
        self.C7 = Conv(128, 64, kersz=ker_size)
        self.U3 = UpSampling(64)
        self.C8 = Conv(64, 32, kersz=ker_size)
        self.U4 = UpSampling(32)
        self.C9 = Conv(32, 16, kersz=ker_size)
        
        self.pred = torch.nn.Conv2d(16, n_classes, 3, 1, 1)
        
        
        if activation is None:
            # don't apply any activation 
            self.act = torch.nn.Identity(32)
            
        else:
            if activation == 'sigmoid':
                # this is for binary segmentation
                self.act = torch.nn.Sigmoid()
                
            elif activation == 'softmax':
                # this is for multi-class segmentation
                self.act = torch.nn.Softmax(dim=1)
                
            else:
                raise ValueError('Activation should be sigmoid/softmax.')
                
     
        
    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        out = self.pred(O4)
        
        out = self.act(out)
            
        return out


if __name__ == '__main__':
    a = torch.randn(2, 3, 128, 128)
    net = MUNet2(n_classes=8, ker_size=5, activation='softmax')
    out = net(a)
    print('input size:', a.shape)
    print('output size:', out.shape)
    r = out[0,:,0,0]
    print(r, r.sum())
    
    net = MUNet2(n_classes=1, ker_size=5, activation='sigmoid')
    out = net(a)
    print('input size:', a.shape)
    print('output size:', out.shape)

    
