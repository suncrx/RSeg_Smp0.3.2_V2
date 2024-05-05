# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:06:18 2022

@author: renxi
"""

## split data into train and val 

import os
import random
import shutil

###################################################################
ROOT_DIR = '/home/chenrx/GeoData/DLData/Waters'
ROOT_DIR = 'D:/GeoData/DLData/vehicle_seg'
ROOT_DIR = 'D:/GeoData/DLData/Saltern/10bands'

SRC_IMG_DIR = os.path.join(ROOT_DIR, 'train', 'images')
SRC_MSK_DIR = os.path.join(ROOT_DIR, 'train', 'masks')
SRC_MSK_DIR2 = os.path.join(ROOT_DIR, 'train', 'masks_2')

VAL_SPLIT = 0.2

MSK_EXT = '.tif'
MSK_EXT2 = '.png'

###################################################################
OUT_DIR = os.path.join(ROOT_DIR, 'val')
DES_IMG_DIR = os.path.join(OUT_DIR, 'images')
DES_MSK_DIR = os.path.join(OUT_DIR, 'masks')
DES_MSK_DIR2 = os.path.join(OUT_DIR, 'masks_2')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DES_IMG_DIR, exist_ok=True)
os.makedirs(DES_MSK_DIR, exist_ok=True)
os.makedirs(DES_MSK_DIR2, exist_ok=True)

imgfiles = os.listdir(SRC_IMG_DIR)

if VAL_SPLIT < 1:
    val_num = int(len(imgfiles)*VAL_SPLIT)
else:
    val_num = int(VAL_SPLIT)

valfiles = random.choices(imgfiles, k=val_num)
count = 0
for imgfn in valfiles:
    mskfn = os.path.splitext(imgfn)[0] + MSK_EXT    
    mskfn2 = os.path.splitext(imgfn)[0] + MSK_EXT2    
    src_msk_path = os.path.join(SRC_MSK_DIR, mskfn)
    src_msk_path2 = os.path.join(SRC_MSK_DIR2, mskfn2)
    if os.path.exists(src_msk_path):
        src_img_path = os.path.join(SRC_IMG_DIR, imgfn)
        
        des_img_path = os.path.join(DES_IMG_DIR, imgfn)
        des_msk_path = os.path.join(DES_MSK_DIR, mskfn)
        des_msk_path2 = os.path.join(DES_MSK_DIR2, mskfn)
        
        shutil.move(src_img_path, des_img_path)
        shutil.move(src_msk_path, des_msk_path)
        shutil.move(src_msk_path2, des_msk_path2)
        
        print(src_img_path, ' -> ', des_img_path)
        print(src_msk_path, ' -> ', des_msk_path)        
        print(src_msk_path2, ' -> ', des_msk_path2)        
        count = count + 1

print('Saved: ', OUT_DIR, '#', count)
print('Done!')

