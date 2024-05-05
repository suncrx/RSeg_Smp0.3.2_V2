# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 19:56:27 2022

@author: renxi

This script converts integer label images to RGB label images.
This script needs a classes.csv file that maps integer to rgb color as follows:

class_id, class_name, R, G, B
0,	unlabeled, 0, 0, 0	
1,	road, 192, 192, 192	
2,	building, 255, 128, 0	
3,	water, 0, 0, 255
4,	vegetation, 0, 255, 0	
5,	land, 128, 64, 0

The first column is the class label (integer label), the second column the
class name. The columns from 3 to 5 are the corresponding RGB values.

"""

import os, sys
import getopt
import cv2


from colorlabel import label_to_rgbLabel, load_rgb_label


#############################################################
# parse arguments from command line

ROOT_DIR = 'D:\\GeoData\\DLData\\AerialImages'
CLS_DEF_FILE = 'classes.csv'

opt, args = getopt.getopt(sys.argv[1:], ["r"], ['root_dir='])
print(opt)
for op, value in opt:
    # root_dir
    if op in ("-r", "--root_dir"):
        ROOT_DIR = value
   
#############################

#load class define and colors
cls_def_file = os.path.join(ROOT_DIR, CLS_DEF_FILE)
class_labels = load_rgb_label(cls_def_file)


out_dir = os.path.join(ROOT_DIR, 'train', 'masks_to_rgbmasks')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# test model on test images
mask_dir = os.path.join(ROOT_DIR, 'train', 'masks')
maskfiles = os.listdir(mask_dir)
for fn in maskfiles:
    fpath = os.path.join(mask_dir, fn)
    print('Converting '+fpath)
    
    mask = cv2.imread(fpath)
    
    rgb_labimg = label_to_rgbLabel(mask, class_labels)
    rgb_labimg = cv2.cvtColor(rgb_labimg, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(os.path.join(out_dir, fn+'.png'), rgb_labimg)
    
print('Done!')    
