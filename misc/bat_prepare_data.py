# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 16:56:07 2021

@author: renxi

This script is used to create train data from each Tile folder.
Images and masks are stored in folders like Tile*, where 'images' folder contains
rgb-images  and 'masks_rgb2' colored label images.

This script will create a 'train' folder with three subfolders, namely 'images', 
'masks', and 'masks_rgb2'. The cropped image patches and the corresponding cropped 
colored label image patches are stored in 'images' and 'masks_rgb2' respectively.
The 'masks' folder contains interger label images with only one channel, which 
indicates each class with an interger, for example: 0 for unlabeld, 1 for road,
2 for building, and so on.

"""


import os, sys
import getopt
import shutil
import numpy as np
import cv2
import random


from patchify import patchify
from PIL import Image


from colorlabel import load_rgb_label, rgbLabel_to_label 


# patchify an image into patches
def patchify_image(imgpath, patch_size=256):
    #Read each image as BGR
    image = cv2.imread(imgpath, 1)  
    #convert to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
    SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
    
    image = Image.fromarray(image)
    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    image = np.array(image)             
   
    #Extract patches from each image
    #print("Now patchifying image:", path+"/"+image_name)
    #Step=256 for 256 patches means no overlap
    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  

    # store each patch in a list
    patches = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = np.squeeze(single_patch_img)
            #Use minmaxscaler instead of just dividing by 255. 
            #single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            
            #single_patch_img = (single_patch_img.astype('float32')) / 255. 
            #single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
            
            patches.append(single_patch_img)

    return patches


def create_data_folders(root_dir, subfolder='train'):
    #create output folder for 'train' or 'test' data
    data_dir = os.path.join(root_dir, subfolder)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    
    #create output folder for croped images
    img_dir = os.path.join(data_dir, 'images')       
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)
    
    #create output folder for croped maskes
    msk_dir = os.path.join(data_dir, 'masks')       
    if os.path.exists(msk_dir):
        shutil.rmtree(msk_dir)
    os.mkdir(msk_dir)
    
    #create output folder for croped rgb-maskes
    mskrgb_dir = os.path.join(data_dir, 'masks_rgb2')       
    if os.path.exists(mskrgb_dir):
        shutil.rmtree(mskrgb_dir)
    os.mkdir(mskrgb_dir)
    
    return data_dir, img_dir, msk_dir, mskrgb_dir

    
# make cropped train images and masks. 
def make_train_test_data(root_dir, patch_size=256, test_percent=0.1,
                         img_ext='.jpg', msk_ext='.png'):        
    
    CLS_FILE = 'classes.csv'  
    # load label information from file
    label_info = load_rgb_label(os.path.join(root_dir, CLS_FILE))
    
    #create output folder for train images
    train_dir, img_dir, msk_dir, mskrgb_dir = create_data_folders(root_dir, 
                                                                  subfolder='train')
    
    result = os.listdir(root_dir)
    #print(result)
    
    nTotal_Patches = 0
    
    img_names = []
    msk_names = []
    mskrgb_names = []
    for itm in result:
        datadir = os.path.join(root_dir, itm) 
        if os.path.isdir(datadir) and itm.lower().startswith('tile'):
            print('Processing '+itm+'...')
            mskdir = os.path.join(datadir, 'masks_rgb2')
            if not os.path.exists(mskdir):
                print('Error:'+mskdir+' does not exist.')
                continue
            imgdir = os.path.join(datadir, 'images')
            if not os.path.exists(imgdir):
                print('Error:'+imgdir+' does not exist.')
                continue
            
            mskfiles = os.listdir(mskdir)
            for mskfn in mskfiles:
                if not mskfn.lower().endswith(msk_ext):
                    continue
                #find the corresponding jpg file.
                imgfn =  os.path.splitext(mskfn)[0]+img_ext
                
                mskpath = os.path.join(mskdir, mskfn)
                imgpath = os.path.join(imgdir, imgfn)
                if (not os.path.exists(imgpath)) or (not os.path.exists(mskpath)):
                    continue
                
                print('Patchifying ...')
                print(imgpath)
                print(mskpath)
                #get image patches
                img_patches = patchify_image(imgpath, patch_size)
                #get mask image patches
                rgb_mask_patches = patchify_image(mskpath, patch_size)    
                
                for img, rgb_msk in zip(img_patches, rgb_mask_patches):
                    # convert rgb-label image to interger label image
                    msk = rgbLabel_to_label(rgb_msk, label_info)
                    
                    img_name = '%08d.jpg'%nTotal_Patches
                    msk_name = '%08d.png'%nTotal_Patches
                    mskrgb_name = '%08d.png'%nTotal_Patches
                    img_names.append(img_name)
                    msk_names.append(msk_name)
                    mskrgb_names.append(mskrgb_name)
                    
                    outimg_path = os.path.join(img_dir, img_name)
                    outmsk_path = os.path.join(msk_dir, msk_name)
                    outmskrgb_path = os.path.join(mskrgb_dir, mskrgb_name)
                    cv2.imwrite(outimg_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(outmskrgb_path, cv2.cvtColor(rgb_msk, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(outmsk_path, msk)                    
                    
                    nTotal_Patches += 1
    print('Total image patches: ', nTotal_Patches)
    
    #split images into train set and test set
    print('Splitting test images ...')
    n_testimgs = int(nTotal_Patches*test_percent)
    # nTotal_Patches = 1300
    # n_testimgs = 120
    idxs = np.int32(np.linspace(0, nTotal_Patches-1, nTotal_Patches))
    idxs = random.choices(idxs.tolist(), k=n_testimgs)
    
    #create output folders for test data
    test_dir, timg_dir, tmsk_dir, tmskrgb_dir = create_data_folders(root_dir, subfolder='test')
    
    #move seletced images from train folder to test folder
    for i in idxs:
        img_path = os.path.join(img_dir, img_names[i])
        msk_path = os.path.join(msk_dir, msk_names[i])
        mskrgb_path = os.path.join(mskrgb_dir, mskrgb_names[i])
        
        timg_path = os.path.join(timg_dir, img_names[i])
        tmsk_path = os.path.join(tmsk_dir, msk_names[i])
        tmskrgb_path = os.path.join(tmskrgb_dir, mskrgb_names[i])
        
        if os.path.exists(img_path):
            shutil.move(img_path, timg_path) 
        if os.path.exists(msk_path):
            shutil.move(msk_path, tmsk_path)
        if os.path.exists(mskrgb_path):
            shutil.move(mskrgb_path, tmskrgb_path)
    
    print('Train image directory: '+train_dir)
    print('Test image directory: '+test_dir)
            

#Read images from repsective 'images' subdirectory
#As all images are of ddifferent size we have 2 options, either resize or crop
#But, some images are too large and some small. Resizing will change the size of real objects.
#Therefore, we will crop them to a nearest size divisible by 256 and then 
#divide all images into patches of 256x256x3. 
def get_data(root_directory, patch_size=256):
    
    # processing rgb image data
    image_dataset = []  
    for path, subdirs, files in os.walk(root_directory):
        #print(path)  
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':   #Find all 'images' directories
            print('Processing '+path)
            images = os.listdir(path)  #List of all image names in this subdirectory
            for i, image_name in enumerate(images):  
                if image_name.endswith(".jpg"):   #Only read jpg images...
                    image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                    SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                    SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                    image = Image.fromarray(image)
                    image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    image = np.array(image)             
           
                    #Extract patches from each image
                    #print("Now patchifying image:", path+"/"+image_name)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            
                            single_patch_img = patches_img[i,j,:,:]
                            
                            #Use minmaxscaler instead of just dividing by 255. 
                            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                            image_dataset.append(single_patch_img)
                    
      
                    
      
    # Now do the same as above for masks
    # For this specific dataset we could have added masks to the above code as masks have extension png
    # processing mask files
    mask_dataset = []  
    for path, subdirs, files in os.walk(root_directory):
        #print(path)  
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'masks_rgb2':   #Find all 'images' directories
            print('Processing '+path)
            masks = os.listdir(path)  #List of all image names in this subdirectory
            for i, mask_name in enumerate(masks):  
                if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
                   
                    mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as Grey (or color but remember to map each color to an integer)
                    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                    SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                    #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                    mask = np.array(mask)             
           
                    #Extract patches from each image
                    #print("Now patchifying mask:", path+"/"+mask_name)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
            
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            
                            single_patch_mask = patches_mask[i,j,:,:]
                            #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                            single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                            mask_dataset.append(single_patch_mask) 
     
    image_dataset = np.array(image_dataset)
    mask_dataset =  np.array(mask_dataset)
    
    return image_dataset, mask_dataset



if __name__ == '__main__':
    #############################################################

    #ROOT_DIR = 'D:\\GeoData\\TestData\\DLData\\segAerialImages'
    
    ROOT_DIR = 'D:\GeoData\TestData\DLData\Buildings'
    ROOT_DIR = 'D:\GeoData\Benchmark\AerialBuildings'
    
    PATCH_SIZE = 256
    IMG_EXT = '.tif'
    MSK_EXT = '.tif'

    #############################################################
    # parse arguments from command line
    opt, args = getopt.getopt(sys.argv[1:], ["r", "p"], ['root_dir=', 'patch_size='])
    print(opt)
    for op, value in opt:
        # root_dir
        if op in ("-r", "--root_dir"):
            ROOT_DIR = value
        if op in ("-p", "--patch_size"):
            PATCH_SIZE = int(value)
            
    #imgs, masks = get_data(ROOT_DIR, PATCH_SIZE)                    
    make_train_test_data(ROOT_DIR, PATCH_SIZE, test_percent=0.05, 
                         img_ext=IMG_EXT, msk_ext=MSK_EXT)