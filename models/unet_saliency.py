# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:02:14 2024

@author: renxi
"""

import torch
import segmentation_models_pytorch as smp


class Unet_Saliency(smp.Unet):
    def __init__(self, **kwargs):
        #print(kwargs)
        super(Unet_Saliency, self).__init__(**kwargs)  
        
        #remove layer4 from the encoder
        #del self.encoder.layer4
        #self.encoder.layer4 = None


if __name__ == '__main__':
    m =Unet_Saliency(encoder_name='resnet18', encoder_depth=3,
                     decoder_channels=[256,128,64],   #all channels (256, 128, 64, 32, 16)
                     )
    print(m)         
    
    d = torch.rand(4, 3, 256,256)
    print(d.shape)
    o=m(d)
    print(o.shape)