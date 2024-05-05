# import the necessary packages

import torch
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import functional as F
from torchvision.transforms import CenterCrop



class Block(Module):
    
    def __init__(self, inChannels, outChannels):
        super().__init__()
        kersz = 3
        # store the convolution and RELU layers
        self.conv1 = Conv2d(inChannels, outChannels, kersz)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kersz)
        
    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))



class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        blockOutputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                 for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
                 for i in range(len(channels) - 1)])
    def forward(self, x, encFeatures):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x
    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures



class MUNet1(Module):
    def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16),
                 n_classes=1, activation=None):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], n_classes, 1)
        #self.retainDim = retainDim
        self.outSize = None
        
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
        N, C, H, W = x.shape
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        seg_map = self.head(decFeatures)
        
        #do activation 
        seg_map = self.act(seg_map)
        
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        #if self.outSize is not None:
        #    seg_map = F.interpolate(seg_map, self.outSize)
        seg_map = F.interpolate(seg_map, (H, W))
        
        # return the segmentation map
        return seg_map

    
if __name__ =='__main__':
    m = MUNet1(n_classes = 6, activation='softmax')
    #print(m)
    inp = torch.randn((1,3,128,256))
    out = m(inp)
    print(out.shape)