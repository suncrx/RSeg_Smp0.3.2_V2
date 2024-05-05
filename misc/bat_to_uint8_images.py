# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:51:38 2023

@author: renxi
"""

import os
import numpy as np
import rasterio

#calculate max, min, and avg values on all images
def st_info(img_dir):
    img_files = os.listdir(IMG_DIR)

    sts = []
    for fn in img_files:
        fpath = os.path.join(IMG_DIR, fn)
        print('Checking '+fpath)
        with rasterio.open(fpath) as db:
            img = db.read()
            nbands = db.count
            sarr = np.zeros((nbands, 3)) 
            for b in range(nbands):
                m = img[b]
                sarr[b,0],sarr[b,1],sarr[b,2] = m.max(), m.min(), m.mean()  
                #print('band %d: max %.3f, min %.3f, avg: %.3f' % (b+1, maxv, minv, avgv) )
        sts.append(sarr)

    stsm = np.stack(sts)    
    st_max = stsm[:,:,0].max(axis=0)
    st_min = stsm[:,:,1].min(axis=0)
    st_avg = stsm[:,:,2].mean(axis=0)
    print('maxv on bands:', st_max)
    print('minv on bands:', st_min)    
    print('avgv on bands:', st_avg)    
    
    return st_max, st_min, st_avg



###############################################################
#ROOT_DIR = 'D:/GeoData/DLData/Waters/WaterDataset/val'
IMG_DIR = 'D:/GeoData/DLData/Saltern/10bands/images'

# calculate statistics
st_max, st_min, st_avg = st_info(IMG_DIR)
nbands = st_max.size

Pdir = os.path.dirname(IMG_DIR)
Odir = os.path.join(Pdir, 'images_3bands')
os.makedirs(Odir, exist_ok=True)

Odir2 = os.path.join(Pdir, 'images_%dbands' % nbands)
os.makedirs(Odir2, exist_ok=True)

img_files = os.listdir(IMG_DIR)
for fn in img_files:
    fpath = os.path.join(IMG_DIR, fn)
    print('Converting '+fpath)
    with rasterio.open(fpath) as db:
        img = db.read()
        nbands = db.count
        h, w = db.height, db.width
        bimg = np.zeros((nbands, h, w), dtype=np.uint8) 
        for b in range(nbands):
            bimg[b,:,:] = np.uint8(255.0*img[b]/st_max[b])
    
    bname = os.path.basename(fn)
    bname, ext = os.path.splitext(bname)
    
    #save all bands
    opath = os.path.join(Odir2, bname+'.tif')
    with rasterio.open(opath, 'w', driver='GTiff',
                   width=w, height=h,
                   count = nbands,
                   dtype = rasterio.dtypes.ubyte) as dst:
        dst.write(bimg)
    
    #save 3 bands
    img_b3 = bimg[0:3,:,:]
    opath = os.path.join(Odir, bname+'.jpg')
    with rasterio.open(opath, 'w', driver='jpeg',
                   width=w, height=h,
                   count = 3,
                   dtype = rasterio.dtypes.ubyte) as dst:
        dst.write(img_b3)

print('Done!')    
