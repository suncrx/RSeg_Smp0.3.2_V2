# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:18:19 2022

@author: renxi
"""
import os, sys
import numpy as np
import csv
import cv2
import matplotlib.pylab as plt

COLORS = [(0,0,0),
(255,255,255),
(255,0,0),
(0,255,0),
(0,0,255),
(255,255,0),
(0,255,255),
(255,0,255),
(192,192,192),
(128,128,128),
(128,0,0),
(128,128,0),
(0,128,0),
(128,0,128),
(0,128,128),
(0,0,128),
]

# generate class information using color list and random colors
def generate_rgb_label(n_classes=32):
    label_info = []    
    for id in range(n_classes):
        CLS_ID = id
        CLS_NAME = 'class%d' % id
        if id<len(COLORS):
            COLOR = COLORS[id]
        else:
            COLOR = np.uint8(255*np.random.rand(3))
        #print(CLS_ID, CLS_NAME, COLOR)
        label_info.append((CLS_ID,CLS_NAME,COLOR))
    return label_info

       
# load rgb label from file
# return a list class_labels, each item in which is a tuple that looks like:
# (id, class_name, [R,G,B])
def load_rgb_label(csv_file_path):
    if not os.path.exists(csv_file_path):
        print(csv_file_path+" does not exist.")
        return None
    file = open(csv_file_path, 'r')
    reader = csv.reader(file, delimiter=',')
    #skip the first line
    next(reader)
    #print('\nClass ID, Class Name, (R,G,B)')
    class_labels = []    
    for line in reader:
        arr = np.array(line)        
        CLS_ID = np.uint8(arr[0])
        CLS_NAME = arr[1]
        COLOR = np.uint8(arr[2:5])
        #print(CLS_ID, CLS_NAME, COLOR)
        class_labels.append((CLS_ID,CLS_NAME,COLOR))
    return class_labels


# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# if matches then replace all values in that pixel with a specific integer
def rgbLabel_to_label(rgb_label_image, class_labels):
    if class_labels is None:
        print('ERROR: class_labels should not be None.')
        sys.exit(0)
        
    img32 = np.uint32(rgb_label_image)
    label_seg = np.zeros(rgb_label_image.shape[0:2], dtype=np.uint8)
    img = (img32[:,:,0]<<16)+(img32[:,:,1]<<8)+img32[:,:,2]
    for idx, item in enumerate(class_labels):
        CLR = (item[2][0]<<16)+(item[2][1]<<8)+item[2][2]
        ind = np.where(img==CLR)
        label_seg[ind]=idx
    return label_seg        


#convert interger label image to RGB label image
def label_to_rgbLabel(label_image, class_labels):
    if len(label_image.shape)>2:
        label_image = label_image[:,:,0]
        
    unqvals = np.unique(label_image)
    max_v = np.max(np.array(unqvals))

    R = np.zeros(label_image.shape[0:2], dtype=np.uint8)
    G = R.copy()
    B = R.copy()
    for cls_id in unqvals:
        idx = np.where(label_image==cls_id)
        #if label_info exists, make a RGB label image 
        #according the class_labels.
        if class_labels is not None:
            if cls_id < len(class_labels):
                R[idx] = class_labels[cls_id][2][0]
                G[idx] = class_labels[cls_id][2][1]
                B[idx] = class_labels[cls_id][2][2]
            else:
                #if cls_id exceeds the total number of colors...
                R[idx] = np.random.randint(0,255)
                R[idx] = np.random.randint(0,255)
                R[idx] = np.random.randint(0,255)
        #if label_info is None, make a gray label image that has three channels. 
        else:
            scalev = np.uint8((cls_id*255.0/(max_v+1)))
            R[idx] = scalev
            G[idx] = scalev
            B[idx] = scalev
    
    rgb_labimg = np.stack([R,G,B], axis=2)
    return rgb_labimg

#display colors of the labels
def show_rgb_labels(class_labels):
    MRG = 32
    SPS = 16
    H, W = 16, 16
    n_labs = len(class_labels)
    img = np.zeros(((H+SPS)*n_labs+2*MRG, 2*MRG+W+128, 3), dtype=np.uint8)
    for i, lab in enumerate(class_labels):
        r = int(lab[2][0])
        g = int(lab[2][1])
        b = int(lab[2][2])
        clr = (r,g,b)
        x, y = MRG, i*(SPS+H)+MRG
        cv2.rectangle(img, (x, y), (x+W, y+H), clr, thickness=10)
    plt.imshow(img)
    

if __name__ == '__main__':
    labs = generate_rgb_label(n_classes=16)
    show_rgb_labels(labs)