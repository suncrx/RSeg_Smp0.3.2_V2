
# import the necessary packages
import os, sys
import random
import rasterio
from rasterio.enums import Resampling

import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from dataug import get_train_aug, get_val_aug

'''
torch.utils.data.Dataset is an abstract class representing a dataset. 
Your custom dataset should inherit Dataset and override the following 
methods:

    __len__ : so that len(dataset) returns the size of the dataset.

    __getitem__: to support the indexing such that dataset[i] can be 
                 used to get i-th sample.

'''

# Dataset for segmentation
# return numpy image (C, H, W)
# and numpy mask (H, W)
class SegDatasetX(Dataset):
    def __init__(self, root_dir, mode="train", 
                 n_classes=1,                  
                 imgH=None, imgW=None,
                 channel_indice=None,
                 #preprocess=None,
                 apply_aug=False, sub_size=-1):
        
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.root = root_dir
        self.imgW = imgW
        self.imgH = imgH        
        
        # binary segmentation : n_classes = 1
        # multi-class segmentation : n_classes > 1
        self.n_classes = n_classes
        
        self.channel_indice = channel_indice
        
        #self.preprocess = preprocess
        
        if apply_aug:
            if mode == 'train':
                self.aug = get_train_aug(height=imgH, width=imgW)
            else:
                self.aug = get_val_aug(height=imgH, width=imgW)
        else:
            self.aug = None
        
        # search image and mask filepaths
        self.images_directory = os.path.join(self.root, mode, "images")
        self.masks_directory = os.path.join(self.root, mode, "masks")
        if not os.path.exists(self.images_directory):
            print("ERROR: Cannot find directory " + self.images_directory)
            sys.exit()
            
        print('Scanning files in %s ... ' % self.mode)
        print(' ' + self.images_directory)
        print(' ' + self.masks_directory)        
        self.imgPairs = self._list_files()
        
        #subset the dataset
        #randomly select num items
        if sub_size > 0 and sub_size <= 1:
            num = np.int32(len(self.imgPairs)*sub_size)
            self.imgPairs = random.sample(self.imgPairs, num)
        elif sub_size > 1:
            num = min(len(self.imgPairs), np.int32(sub_size))
            self.imgPairs = random.sample(self.imgPairs, num)
        
        print(" #image pairs: ", len(self.imgPairs))



    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imgPairs)


    # return a tuple  (image, mask)
    # image: tensor image with shape (C, H, W), and data range (0 ~ 1.0)
    # mask: binary mask image of size (H, W), with value 0 and 1.0.
    def __getitem__(self, idx):
        # grab the image and mask path from the current index
        imagePath = self.imgPairs[idx]['image']
        maskPath = self.imgPairs[idx]['mask']
                
        # 1)Reading image using rasterio package
        nbands = 3
        with rasterio.open(imagePath) as imgd:
            # image stored in format [C, H, W]
            if self.imgH and self.imgW:
                if self.imgH != imgd.height or self.imgW != imgd.width:
                    image = imgd.read(out_shape=(imgd.count, self.imgH, self.imgW),
                                     resampling=Resampling.bilinear)
                else:
                    image = imgd.read()
            else:
                image = imgd.read()
            nbands = imgd.count
            # extract the selected bands    
            if self.channel_indice:
                image = image[self.channel_indice]
                nbands = len(self.channel_indice)
            # transpose to [H,W,C]    
            image = image.transpose(1,2,0)                
            
                            
        # 2)Reading the associated mask from disk in grayscale mode
        if os.path.exists(maskPath):
            #mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            with rasterio.open(maskPath) as mskd:
                if self.imgH and self.imgW:
                    if self.imgH != mskd.height or self.imgW != mskd.width:
                        mask = mskd.read(out_shape=(mskd.count, self.imgH, self.imgW),
                                         resampling=Resampling.nearest)
                    else:
                        mask = mskd.read()                        
                else:
                    mask = mskd.read()
            mask = mask[0]                        
        else:
            mask = np.zeros(image.shape[0:2], dtype=np.uint8)

        
        # apply augmentation (augmentation is only for 1-band and 3-band images)
        if self.aug and (nbands==3 or nbands==1):
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
       
        # apply preprocessing on image (not on mask)
        #if self.preprocess:
        #    image = self.preprocess(image)    
       
        # [1] convert image to tensor of shape [C, H, W], with 
        #     values between(0, 1.0)
        if image.dtype == 'uint8':
            image = image/255.0;        
        image = transforms.ToTensor()(image).float()
        
        # [2] transform mask to tensor
        # binary segmentation            
        if self.n_classes < 2:            
            # convert to (0, 1) float 
            mask = transforms.ToTensor()(mask > 0).float()
            return (image, mask)                        
        # multi-class segmentation
        else:           
            # convert label mask to one-hot tensor
            mask = torch.squeeze(transforms.ToTensor()(mask).long())
            mask_onehot = F.one_hot(mask, num_classes=self.n_classes)
            mask_onehot = torch.transpose(mask_onehot, 2, 0)
            return (image, mask_onehot)
            
     
    def get_image_and_mask_path(self, idx):
        return (self.imgPairs[idx]['image'], self.imgPairs[idx]['mask'])
    
    
    # get all image filepaths   
    def get_image_filepaths(self):
        return [item['image'] for item in self.imgPairs]
    
    
    def _list_files(self):
        #EXTS = ['.png', '.bmp', '.gif', '.jpg', '.jpeg']
        imgs = os.listdir(self.images_directory)
        if os.path.exists(self.masks_directory):
            msks = os.listdir(self.masks_directory)
        else:
            msks = []
        
        #extract mask file names and extensions
        msk_names = []
        msk_exts = []
        for i in range(len(msks)):
            path = os.path.join(self.masks_directory, msks[i])
            if not os.path.isfile(path):
                continue
            fname, ext = os.path.splitext(msks[i])
            msk_names.append(fname)
            msk_exts.append(ext)
            
        # extract image and mask pairs        
        imgpaths = []
        mskpaths = []
        for i in range(len(imgs)):
            path_img = os.path.join(self.images_directory, imgs[i])
            if not os.path.isfile(path_img):
                continue
            
            fname, ext = os.path.splitext(imgs[i])
            # if finding a matched mask file in msk_names
            if fname in msk_names:
                idx = msk_names.index(fname)
                path_msk = os.path.join(self.masks_directory, fname + msk_exts[idx])
            # or not, generate a virtual filepath for the mask file
            else:
                path_msk = os.path.join(self.masks_directory, fname + '.png')
                print('Warning: Cannot find mask file %s' % path_msk)
            
            imgpaths.append(path_img)
            mskpaths.append(path_msk)                        
        
        #make image pairs list
        imgPairs = [{'image':fp1, 'mask':fp2} for fp1, fp2 in zip(imgpaths, mskpaths)]    
        
        return imgPairs                                  


# check data integrity
def check_data(root_dir):
    subdirs = ['train', 'val']
    for sd in subdirs:
        img_dir = os.path.join(root_dir, sd, 'images') 
        msk_dir = os.path.join(root_dir, sd, 'masks') 
    
        if not os.path.exists(img_dir):
            print('ERROR: '+img_dir+' does not exist.')
            return False
        if not os.path.exists(msk_dir):
            print('ERROR: '+msk_dir+' does not exist.')
            return False

    return True    

    
if __name__ == '__main__':
    import matplotlib.pylab as plt

    #data_dir = 'D:\\GeoData\\DLData\\buildings'   
    data_dir = 'D:\\GeoData\\DLData\\Saltern\\10bands'    
    #data_dir = 'D:\\GeoData\\DLData\\AerialImages'      
    
    ds = SegDatasetX(data_dir, 'train', n_classes=1, 
                     imgH=128, imgW=128,
                     band_indice=[0, 1, 9],
                    apply_aug=False, sub_size=32)        
    for i in range(10):        
        samp = ds[i]
        img, msk = samp
        
        #print('original image: ', oimg.shape, omsk.shape)
        print('transformed image: ', img.shape, msk.shape)
        
        plt.figure()
        #plt.subplot(221)        
        #plt.imshow(img)        
        #plt.subplot(222)        
        #plt.imshow(msk)                
        plt.subplot(121)        
        tmpimg = img[0:3].numpy()
        plt.imshow(np.moveaxis(tmpimg, 0, -1))        
        plt.subplot(122)        
        plt.imshow(torch.squeeze(msk).numpy())             
        plt.show()
