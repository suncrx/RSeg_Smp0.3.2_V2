# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 10:15:17 2021

@author: renxi

This script converts rgb label images from one style to another style.
this script needs a csv file that maps one RGB color to another RGB color,
such as:

R1,G1,B1, R2, G2, B2
132, 41, 246, 0, 0, 0	
110, 193, 228, 192, 192, 192	
60, 16, 152, 255, 128, 0	
226, 169, 41, 0, 0, 255
254, 221, 58, 0, 255, 0	
155,155,155, 128, 64, 0    

where, each line contains the RGB components of two colors. The first color1 (R1 G1 B1) is 
the source color and the second color2 (R2 G2 B2) the target. The script will replace (R1 G1 B1) in
all the mask images with (R2 G2 B2). 

"""

import os, sys
import imutils
import csv
import shutil
import getopt
import skimage
import numpy as np

###############################################################
ROOT_DIR = 'D:\\GeoData\\DLData\\AerialImages'
IN_FOLDER = 'masks_rgb'
OUT_FOLDER = 'masks_rgb2'
#color transfer file
color_trans_file = 'color_trans1.csv'



###############################################################

# parse arguments from command line
opt, args = getopt.getopt(sys.argv[1:], ["r"], ['root_dir='])
print(opt)
for op, value in opt:
	# root_dir
	if op in ("-r", "--root_dir"):
		ROOT_DIR = value
        
        

# load colors from the csv file
csvpath = os.path.join(ROOT_DIR, color_trans_file)
if not os.path.exists(csvpath):
    print(csvpath+" does not exist.")
    sys.exit(0)
COLOR_LIST1 = []
COLOR_LIST2 = []    
file = open(csvpath, 'r')
reader = csv.reader(file, delimiter=',')
#skip the first line
next(reader)
for line in reader:
    arr = np.array(line).astype(np.uint8)
    COLOR_LIST1.append(arr[0:3])
    COLOR_LIST2.append(arr[3:6])
print('Colors 1:', COLOR_LIST1)    
print('Colors 2:', COLOR_LIST2)    


#converting begins    
result = os.listdir(ROOT_DIR)
#print(result)
for itm in result:
    datadir = os.path.join(ROOT_DIR, itm) 
    if os.path.isdir(datadir) and itm.lower().startswith('tile'):
        print('Processing '+itm+'...')
        maskdir = os.path.join(datadir, IN_FOLDER)
        if not os.path.exists(maskdir):
            print('Error:'+maskdir+' does not exist.')
            continue
        
        #create 'mask2_rgb' folder for output
        outdir = os.path.join(datadir, OUT_FOLDER)       
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        
        #read each mask image, convert and save to mask2_rgb folder.
        mskfiles = os.listdir(maskdir)
        for fn in mskfiles:
            if not fn.lower().endswith('png'):
                continue
            img = skimage.io.imread(os.path.join(maskdir, fn))[:,:,0:3].astype(np.int32)
            img2 = (img[:,:,0]<<16)+(img[:,:,1]<<8)+img[:,:,2]
            R = np.zeros(img2.shape, dtype=np.uint8)
            G = np.zeros(img2.shape, dtype=np.uint8)
            B = np.zeros(img2.shape, dtype=np.uint8)
            for n, CLR in enumerate(COLOR_LIST1):
                CLRV = (CLR[0]<<16)+(CLR[1]<<8)+CLR[2]
                ind = np.where(img2==CLRV)
                R[ind] = COLOR_LIST2[n][0]
                G[ind] = COLOR_LIST2[n][1]
                B[ind] = COLOR_LIST2[n][2]
            img = np.stack([R,G,B], axis=2)    
            outpath = os.path.join(outdir, fn)
            skimage.io.imsave(outpath, img)
            
print('Done!')            

