#Squeeze and excitation layer

import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    se64 = SELayer(64)
    se128 = SELayer(128)
    print(se64)
    print(se128)
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', 
                           pretrained=True)
    layer1 = model.layer1 
    print(model.layer1)
    
    #add se layers to layer1
    model.layer1 = nn.Sequential(layer1[0], se64, 
                                 layer1[1], se128)
    print(model.layer1)
    
    model.fc = nn.Linear(512, 6)
    print(model)
    
    x = torch.rand(4, 3,128,128)
    o = model(x)
    
    
    