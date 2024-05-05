# USAGE
# python test.py
#

# import the necessary packages
# %% import installed packages
import os
import sys
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pylab as plt

import segmentation_models_pytorch as smp
# explictly import utils if segmentation_models_pytorch >= 0.3.2
from segmentation_models_pytorch import utils as smp_utils 

import models
from datasetx import SegDatasetX
from misc.plot import plot_prediction


#%% get current directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # FasterRCNN root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# determine the device to be used for training and evaluation
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print('Device : ', DEV)


#======================================================================
#padding image to the size of 32*X
#The shape of input image should be (Channel, Height, Width), e.g. 3*213*2455
def pad_image_32x(image):
    c, h, w = image.shape
    if 32*int(w/32) != w:
        padx = 32*int(w/32+1)-w
        image = F.pad(image, (0,padx,0,0,0,0))
    if 32*int(h/32) != h:
        pady = 32*int(h/32+1)-h
        image = F.pad(image, (0,0,0,pady,0,0))
    return image
        
#======================================================================
def make_prediction(model, image, out_H, out_W, binary=False, conf=0.5):  
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad(): 
        #padding image size to 32*M
        c, h, w = image.shape
        image = pad_image_32x(image)
        
        # apply image transformation. This step turns the image into a tensor.
        # with the shape (1, 3, H, W). See IMG_TRANS in dataset.py
        image = torch.unsqueeze(image, 0)        
        # make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        pred = model.forward(image).squeeze()                

        # crop prediction size to the original size
        pred = pred[0:h,0:w]
        
        # Sigmod or softmax has been performed in the net
        if binary:
            pred = cv2.resize(pred.numpy(), (out_W, out_H))
            pred = np.uint8(pred>=conf)
            #pred = np.uint8(pred*255)            
        else:
            #determine the class by the index with the maximum                     
            pred = np.uint8(torch.argmax(pred, dim=0))        
            #resize to the original size        
            pred = cv2.resize(pred, (out_W, out_H),
                                   interpolation=cv2.INTER_NEAREST)
            #print('Found classes: ', np.unique(pred))                              
    return pred


# calculate the evaluation metrics between the predicted mask and ground-truth
def eval_metrics(predMask, gtMask, n_classes):
    if n_classes>1:
        pm = [predMask==v for v in range(n_classes)]
        gm = [gtMask==v for v in range(n_classes)]
        pm, gm = np.array(pm), np.array(gm)
    else:
        pm, gm = np.uint8(predMask>0), np.uint8(gtMask>0)        
    #iou = smp.utils.metrics.IoU()  #for smp 0.2.1
    iou = smp_utils.metrics.IoU()   #for smp >= 0.3.2
    acc = smp_utils.metrics.Accuracy()
    pre = smp_utils.metrics.Precision()
    rec = smp_utils.metrics.Recall()
    fsc = smp_utils.metrics.Fscore()
    
    iouv = iou.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    accv = acc.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    prev = pre.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    recv = rec.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    fscv = fsc.forward(torch.as_tensor(pm), torch.as_tensor(gm))
    
    #return iouv
    return {'iou':iouv, 'acc':accv, 'pre':prev, 'rec':recv, 'fsc':fscv}


def run(opt):
    #print(opt)
    # get parameters
    data_dir, img_sz = opt.data_dir, opt.img_sz
    model_file, out_dir = opt.model_file, opt.out_dir
    conf, plot = opt.conf, opt.plot
    
    if not os.path.exists(model_file):
        raise Exception('Can not find model path: %s' % model_file)
    
    # make output folders
    model_basename = os.path.basename(model_file)
    # make output folder for predicting images
    os.makedirs(os.path.join(data_dir, 'out'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'out', 'test_pred'), exist_ok=True)
    outpred_dir = os.path.join(data_dir, 'out', 'test_pred', model_basename)
    if os.path.exists(outpred_dir):
        shutil.rmtree(outpred_dir)
    os.makedirs(outpred_dir)
        
    #--------------------------------------------------------------------    
    # load our model from disk and flash it to the current device
    print("Loading model: %s" % model_file)
    (model, model_name, 
     n_classes, class_names, in_channels) = models.utils.load_seg_model(model_file)    
    
    #---------------------------------------------------------------------------
    # load the image paths in our testing directory and
    # randomly select 10 image paths
    print("Loading test image ...")
    #preprocessing function from segmentation-models-pytorch package
    #preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')    
    testDS = SegDatasetX(data_dir, mode="test", 
                        n_classes=n_classes, 
                        imgH=img_sz, imgW=img_sz,
                        #preprocess=preprocessing_fn,
                        apply_aug = False)
    
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    
    nm = len(testDS)
    #nm = min(len(testDS), 10)
    metrics = []
    imgnames = []
    for i in range(nm):
        #get preprocessed image and mask
        img, gtMask =  testDS[i]            
        #get the image and mask filepaths
        imgPath, mskPath = testDS.get_image_and_mask_path(i)        
        
        #get the original image and mask
        ori_img = np.uint8(img.numpy()[0:3]*255).transpose(1,2,0)
        ori_gtMask = np.uint8(gtMask.numpy()*255).squeeze()

        outH, outW = ori_img.shape[0:2]
        
        # make predictions and visualize the results
        print('\nPredicting ' + imgPath)    
        #for binary segmentation, pred is a uint8-type mask with 0, 1;
        #for multi-class segmentation, pred is a uint8-type mask with
        #class labels: 0, 1, 2, 3, ... , n_class-1
        is_binary = (n_classes<2)
        pred = make_prediction(model, img, outH, outW, 
                               binary=is_binary, conf=conf)
        
        #evaluation
        #iouv = Cal_IoU(pred, ori_gtMask, n_classes=n_classes)    
        res = eval_metrics(pred, ori_gtMask, n_classes=n_classes)     
        iouv, accv, prev = res['iou'], res['acc'],res['pre']
        recv, fscv = res['rec'], res['fsc']         
        metrics.append([iouv, accv, prev, recv, fscv])
        print('IoU: %.3f Acc: %.3f Prec: %.3f Rec: %.3f Fscore: %.3f' % 
              (iouv, accv, prev, recv, fscv))
        image_basename = os.path.basename(imgPath) 
        imgnames.append(image_basename)
        
        #------------------------------------
        #save and convert results to rgb label for visualization 
        image_basename = os.path.basename(imgPath) 
        bname, ext = os.path.splitext(image_basename)
        out_mskPath = os.path.join(outpred_dir, bname+'_msk.png')        
        fig_path = os.path.join(outpred_dir, 'plot_'+bname+'.png')           
        stitle = '%s IoU %.3f' % (image_basename, iouv)
        #save image
        bgrimg = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(outpred_dir, image_basename), bgrimg)    
        if is_binary:
            #save predicted mask (one channel)    
            Mask = np.uint8(pred*255)        
            cv2.imwrite(out_mskPath, Mask)
            #save gt mask
            if ori_gtMask.max()==1:
                ori_gtMask = ori_gtMask*255
            cv2.imwrite(os.path.join(outpred_dir, bname+'_gt.png'), ori_gtMask)                
            #plot results
            if plot:
                plot_prediction(ori_img, ori_gtMask, Mask, 
                                sup_title=stitle, save_path=fig_path, 
                                auto_close=True)   
        else:
            #save predicted mask
            Mask = np.uint8(pred)        
            cv2.imwrite(out_mskPath, Mask)
            Mask_rgb = Mask        
            out_rgbMskPath = os.path.join(outpred_dir, bname+'_rgb.png')        
            cv2.imwrite(out_rgbMskPath, Mask_rgb)
            #save gt mask            
            ori_gtMask_rgb = ori_gtMask
            cv2.imwrite(os.path.join(outpred_dir, bname+'_gt.png'), ori_gtMask_rgb)
            #plot results
            if plot:
                plot_prediction(ori_img, ori_gtMask_rgb, Mask_rgb, 
                                sup_title=stitle, save_path=fig_path, 
                                auto_close=True)   
        
            '''
            #convert Mask to rgb label image and save
            RgbMask = label_to_rgbLabel(Mask, label_colors)
            BgrMask = cv2.cvtColor(RgbMask, cv2.COLOR_RGB2BGR)            
            cv2.imwrite(out_rgbMskPath, BgrMask)
            print('Saved: %s' % out_rgbMskPath)
            
            #convert ground-truth mask to rgb label image
            gtRgbMask = label_to_rgbLabel(ori_gtMask, label_colors)            
            '''
        
    #Evaluation metrics 
    Mm = np.array(metrics)
    maxv = Mm.max(axis=0)
    minv = Mm.min(axis=0)
    meanv = Mm.mean(axis=0)
    print('\nname,  Iou,  Accuracy,  Precision, Recall, Fscore')
    print('Max,  %.3f, %.3f, %.3f, %.3f, %.3f' % 
          (maxv[0],maxv[1],maxv[2],maxv[3],maxv[4]))
    print('Min,  %.3f, %.3f, %.3f, %.3f, %.3f' % 
          (minv[0],minv[1],minv[2],minv[3],minv[4]))
    print('Mean,  %.3f, %.3f, %.3f, %.3f, %.3f' % 
          (meanv[0],meanv[1],meanv[2],meanv[3],meanv[4]))
    #plt.figure()
    #plt.plot(Mm[0], '.')
    #plt.show()        
    print('Done!')    
    print('Results saved: %s' % outpred_dir)       
    
    #write metrics to log file
    logfn = os.path.join(outpred_dir, model_basename+'_log.txt')
    with open(logfn, 'w') as fo:
        print('\nname,  Iou,  Accuracy,  Precision, Recall, Fscore', file=fo)
        for i in range(Mm.shape[0]):
            print('%s, %.3f, %.3f, %.3f, %.3f, %.3f' % 
                  (imgnames[i],Mm[i][0],Mm[i][1],Mm[i][2],Mm[i][3],Mm[i][4]),
                  file=fo)
        print('%s, %.3f, %.3f, %.3f, %.3f, %.3f' % 
                ('Max',maxv[0],maxv[1],maxv[2],maxv[3],maxv[4]), file=fo)
        print('%s, %.3f, %.3f, %.3f, %.3f, %.3f' % 
                ('Min',minv[0],minv[1],minv[2],minv[3],minv[4]), file=fo)
        print('%s, %.3f, %.3f, %.3f, %.3f, %.3f' % 
                ('Mean',meanv[0],meanv[1],meanv[2],meanv[3],meanv[4]), file=fo)
        

#%% parse arguments from command line
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str, 
                        #default='D:/GeoData/DLData/Waters/WaterTiles/out/unet_resnet34_best.pt', 
                        #default='D:/GeoData/DLData/vehicle_seg/out/600epochs/unet_resnet34_best.pt',                         
                        default='D:/GeoData/DLData/Saltern/10bands/out/unet_resnet18_best.pt',                         
                        help='model filepath')
    
    parser.add_argument('--data_dir', type=str, 
                        #default='D:/GeoData/DLData/Waters/WaterTiles', 
                        #default='D:/GeoData/DLData/vehicle_seg', 
                        default='D:/GeoData/DLData/Saltern/10bands', 
                        help='test image directory')
    
    parser.add_argument('--img_sz', type=int, 
                        default=None, help='input image size (pixels)')
    
    parser.add_argument('--out_dir', type=str, default=ROOT / 'out', 
                        help='training output path')    
    
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='test confidence')    
    
    parser.add_argument('--plot', type=bool, default=True, 
                       help='Plot the results or not')   
               
    return parser.parse_args() 


if __name__ == '__main__':
    opt = parse_opt()
    run(opt)
    