# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:06:18 2022

@author: renxi
"""

## split test data 

import os, sys
import random
import shutil

###################################################################
ROOT_DIR = 'D:\\GeoData\\DLData\\AerialImages'

SRC_IMG_DIR = os.path.join(ROOT_DIR, 'train', 'images')
SRC_MSK_DIR = os.path.join(ROOT_DIR, 'train', 'masks')

TEST_QUANT = 30

MSK_EXT = '.png'

###################################################################
OUT_DIR = os.path.join(ROOT_DIR, 'test3')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
    #print('ERROR: %s exists. Please select another folder.')
    #sys.exit(0)

DES_IMG_DIR = os.path.join(OUT_DIR, 'images')
DES_MSK_DIR = os.path.join(OUT_DIR, 'masks')

#os.mkdir(OUT_DIR)
if not os.path.exists(DES_IMG_DIR):
    os.mkdir(DES_IMG_DIR)
if not os.path.exists(DES_MSK_DIR):    
    os.mkdir(DES_MSK_DIR)


imgfiles = os.listdir(SRC_IMG_DIR)

if TEST_QUANT < 1:
    test_num = int(len(imgfiles)*TEST_QUANT)
else:
    test_num = int(TEST_QUANT)

testimgfiles = random.choices(imgfiles, k=test_num)
count = 0
for imgfn in testimgfiles:
    mskfn = os.path.splitext(imgfn)[0] + MSK_EXT    
    src_msk_path = os.path.join(SRC_MSK_DIR, mskfn)
    if os.path.exists(src_msk_path):
        src_img_path = os.path.join(SRC_IMG_DIR, imgfn)
        
        des_img_path = os.path.join(DES_IMG_DIR, imgfn)
        des_msk_path = os.path.join(DES_MSK_DIR, mskfn)
        
        shutil.move(src_img_path, des_img_path)
        shutil.move(src_msk_path, des_msk_path)
        
        print(src_img_path, ' -> ', des_img_path)
        print(src_msk_path, ' -> ', des_msk_path)        
        count = count + 1

print('Saved: ', OUT_DIR, '#', count)
print('Done!')

