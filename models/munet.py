'''
PyTorch implementation of UNet
'''

import torch
import torch.nn as nn


__all__ = [
    "ConvBlock",
    "UpConv",
    "MUnet"
]
    
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, 
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, 
                      padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

#---------------------------------------------
# Original UNet
class MUNet(nn.Module):
    def __init__(self, in_channels=3,  n_classes=1, 
                 activation=None):
        
        super(MUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(in_channels, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)


        self.Up5 = UpConv(1024, 512)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
        
        self.act = None
        if activation and activation.lower() == 'sigmoid':
            # this is for binary segmentation
            self.act = torch.nn.Sigmoid()
            
        elif activation and activation.lower() == 'softmax':
            # this is for multi-class segmentation
            self.act = torch.nn.Softmax(dim=1)        
        
        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)    


    def forward(self, x):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        # concatenate attention-weighted skip connection with previous layer output
        d5 = torch.cat((e4, d5), dim=1) 
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        # concatenate attention-weighted skip connection with previous layer output        
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)        
        # concatenate attention-weighted skip connection with previous layer output
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        # concatenate attention-weighted skip connection with previous layer output
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)
        
        if self.act is not None:
            out = self.act(out)

        return out


if __name__ == '__main__':
    m = MUNet(in_channels=3, n_classes=5, activation='softmax')
    print(m)
    
    d = torch.rand(1, 3, 256,256)
    print(d.shape)
    o=m(d)
    print(o.shape)