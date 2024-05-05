# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:51:38 2023

@author: renxi
"""

import os, sys
import imutils
import shutil
import numpy as np
import PIL

###############################################################
ROOT_DIR = 'D:/GeoData/DLData/Waters/WaterDataset/val'
ROOT_DIR = 'D:/GeoData/DLData/Saltern/Balunmahai'
ROOT_DIR = 'D:/GeoData/DLData/Saltern/10Bands/train'

IN_FOLDER = 'masks'
OUT_FOLDER = 'masks_2'
OUT_FOLDER_PNG = 'masks_png'

out_dir = os.path.join(ROOT_DIR, OUT_FOLDER)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

mask_dir = os.path.join(ROOT_DIR, IN_FOLDER)
maskfiles = os.listdir(mask_dir)
posn = 0
negn = 0
for fn in maskfiles:
    fpath = os.path.join(mask_dir, fn)
    print('Converting '+fpath)
    mask = PIL.Image.open(fpath)
    mask = PIL.ImageOps.grayscale(mask)
    
    #convert to png mask
    if not fpath.endswith('.png'):
        pngdir = os.path.join(ROOT_DIR, OUT_FOLDER_PNG)
        os.makedirs(pngdir, exist_ok=True)
        bfn, ext = os.path.splitext(fn)
        fppng = os.path.join(pngdir, bfn+'.png')
        mask.save(fppng)
    
    #save 0-255 binary mask
    m = np.array(mask)    
    print(np.unique(m))
    m = np.uint8(m>0)*255   
            
    msk2 = PIL.Image.fromarray(m)
    
    bfn, ext = os.path.splitext(fn)
    out_path = os.path.join(out_dir, bfn+'.png')
    print(out_path)
    msk2.save(out_path)
    

    
    maxv = m.max()
    if maxv == 0:
        negn += 1
    else:
        posn += 1

print('Done!')    
print('Positive samples:', posn)
print('Negative samples:', negn)