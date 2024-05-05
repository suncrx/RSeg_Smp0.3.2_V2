# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 19:06:47 2023

@author: renxi
"""
# import the necessary packages
import torch
import segmentation_models_pytorch as smp

from .munet  import MUNet
from .munet_ag  import MUNet_AG
from .munet_cbam  import MUNet_CBAM

from .munet1 import MUNet1
from .munet2 import MUNet2

from .unet_att import Unet_Attention


#-----------------------------------------------------------------------------
def create_model(arct='unet', encoder='resnet34', encoder_weigths='imagenet',
                 n_classes=1, in_channels=3):

    m_fullname = arct + '_' + encoder     
    
    activation_name = 'softmax' if n_classes>1 else "sigmoid"
    
    ENCODER_WEIGHTS = encoder_weigths
    ENCODER = encoder
    
    if arct.lower()=='unet':
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 
        # use `imagenet` pre-trained weights for encoder initialization
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # model output channels (number of classes in your dataset)
        MODEL = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)
                         #activation='softmax')
                         
    # unet using scse attention                         
    elif arct.lower()=='unet_scse':
        MODEL = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         decoder_attention_type='scse') 
        
    elif arct.lower()=='unet_se':                        
        MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         decoder_attention_type='se')

    elif arct.lower()=='unet_sk':                        
        MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name,
                         decoder_attention_type='sk')
    
    elif arct.lower()=='unet_cbam':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='cbam') 

    elif arct.lower()=='unet_bam':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='bam') 

    elif arct.lower()=='unet_eca':                        
         MODEL = Unet_Attention(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name,
                          decoder_attention_type='eca') 

                         
    elif arct.lower()=='unetplusplus':
        MODEL = smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,     
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)

                         
    elif arct.lower()=='linknet':
        MODEL = smp.Linknet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                         in_channels=in_channels,  classes=n_classes,
                         activation=activation_name)
                         #activation='softmax')
    
    elif arct.lower()=='fpn':
        MODEL = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)
    
    elif arct.lower()=='deeplabv3':
        MODEL = smp.DeepLabV3(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)
        
    elif arct.lower()=='deeplabv3plus':
        MODEL = smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                              in_channels=in_channels,  classes=n_classes,
                              activation=activation_name)  
    
    elif arct.lower()=='manet':
        MODEL = smp.MAnet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                          in_channels=in_channels,  classes=n_classes,
                          activation=activation_name)  

    elif arct.lower()=='pan':
        MODEL = smp.PAN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)     

    elif arct.lower()=='pspnet':
        MODEL = smp.PSPNet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                        in_channels=in_channels,  classes=n_classes,
                        activation=activation_name)              
    
    #-------------------------------------------------------------        
    elif arct.lower()=='munet':    
        MODEL = MUNet(in_channels=in_channels, n_classes=n_classes, 
                      activation=activation_name)
        m_fullname = arct

    elif arct.lower()=='munet_ag':    
        MODEL = MUNet_AG(in_channels=in_channels, n_classes=n_classes, 
                      activation=activation_name)
        m_fullname = arct

    elif arct.lower()=='munet_cbam':    
        MODEL = MUNet_CBAM(in_channels=in_channels, n_classes=n_classes, 
                      activation=activation_name)
        m_fullname = arct
        
    elif arct.lower()=='munet1':        
        MODEL = MUNet1(n_classes=n_classes, activation=activation_name)
        m_fullname = arct
        
    elif arct.lower()=='munet2':        
        MODEL = MUNet2(n_classes=n_classes, activation=activation_name)
        m_fullname = arct

    else:
        MODEL = None
        m_fullname = arct
        
    return MODEL, m_fullname        


# generate a model file name 
def generate_model_filename(model_name):
    mfname = 'seg4_%s_best.pth' % model_name  
    return mfname


# save model and auxillary information
def save_seg_model(fpath, model, arct, encoder, 
                   n_classes, class_names, in_channels):
    torch.save({
            'n_classes': n_classes,
            'class_names': class_names, 
            'in_channels': in_channels,
            'arct': arct,
            'encoder': encoder,
            'model_state_dict': model.state_dict(),                        
            },  fpath)

# save model from model filepath
def load_seg_model(fpath):
    # load the model and the trained weights
    mdict = torch.load(fpath)
    
    n_classes = mdict['n_classes']
    class_names = mdict['class_names']
    in_channels = mdict['in_channels']
    arct = mdict['arct']
    encoder = mdict['encoder']
    
    model, model_name = create_model(arct=arct, encoder=encoder, 
                                     encoder_weigths='imagenet',
                                     n_classes=n_classes, 
                                     in_channels=in_channels)
    model.load_state_dict(mdict['model_state_dict']) #, map_location=DEVICE)
    
    return (model, model_name, n_classes, class_names, in_channels)


# save training check point
def save_checkpoint(fpath, model, arct, encoder, 
               optimizer_name, optimizer,
               n_classes, class_names, n_channels, 
               epochs, batch_size,
               #lr, momentum, weight_decay, 
               train_losses, val_losses, train_scores, val_scores):
    torch.save({
            'n_classes': n_classes,
            'class_names': class_names, 
            'n_channels': n_channels,
            
            'epochs': epochs,
            'batch_size': batch_size,
#            'lr': lr,
#            'momentum': momentum,
#            'weight_decay': weight_decay,
            
            'arct': arct,
            'encoder': encoder,
            'model_state_dict': model.state_dict(),            
 
            'opti_name': optimizer_name,
            'opti_state_dict': optimizer.state_dict(),            

            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_scores': train_scores,
            'val_scores': val_scores,
            },  fpath)
    

    
if __name__ == '__main__':
    m, mname = create_model(arct='unet')
    print(m)
    print(mname)
    mfname = generate_model_filename(mname)
    print(mfname)