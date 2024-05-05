# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:31:56 2024

@author: renxi


Attention Unet 

The Unet is created from segmentation_models_pytorch and then
modified by adding attention modules. 
 
"""

import torch
import segmentation_models_pytorch as smp

'''
import attse
import attsk
import attcbam
import attbam
import atteca

'''

from . import attse
from . import attsk
from . import attbam
from . import attcbam
from . import atteca



class Unet_Attention(smp.Unet):
    def __init__(self, decoder_attention_type = 'se', **kwargs):
        #print(kwargs)
        super(Unet_Attention, self).__init__(**kwargs)  
        
        att_type = decoder_attention_type.lower()
        assert att_type in ['se', 'sk', 'cbam', 'bam', 'eca']
        
        #Adding attention modules. 
        #This can be done by replacing the 
        #Identity modules attention1 and attention2 with SE, CBAM, and ...
        for blk in self.decoder.blocks:
            out_channels = blk.conv1[0].out_channels
            in_channels = blk.conv1[0].in_channels            
            if att_type == 'se':
                att1 = attse.SELayer(in_channels, reduction=8)
            elif att_type == 'cbam':
                att1 = attcbam.CBAM(in_channels, reduction=8)
            elif att_type == 'bam':
                att1 = attbam.BAM(in_channels, reduction=8)                
            elif att_type == 'eca':
                att1 = atteca.ECA(in_channels)                
            elif att_type == 'sk':
                att1 = attsk.SKLayer(in_channels, in_channels)
                    
            blk.attention1 = att1
            
            
            out_channels = blk.conv2[0].out_channels
            in_channels = blk.conv2[0].in_channels            
            if att_type.lower() == 'se':
                att2 = attse.SELayer(in_channels, reduction=8)
            elif att_type.lower() == 'cbam':
                att2 = attcbam.CBAM(in_channels, reduction=8)
            elif att_type == 'bam':
                att2 = attbam.BAM(in_channels, reduction=8)                                
            elif att_type == 'eca':
                att2 = atteca.ECA(in_channels)                
            elif att_type == 'sk':
                att2 = attsk.SKLayer(in_channels, in_channels)
                
            blk.attention2 = att2
            

if __name__ == '__main__':
    
    m = Unet_Attention(encoder_name='resnet34', 
                  in_channels=3, classes=6, 
                  decoder_attention_type='eca')
    #print(un)
    d = torch.rand(4, 3, 256,256)
    print(d.shape)
    o=m(d)
    print(o.shape)
        