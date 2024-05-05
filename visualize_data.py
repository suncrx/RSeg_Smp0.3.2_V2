# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:42:51 2023

@author: renxi
"""

#run
# python visualize_data.py --cfg configs/waters.yaml
# python visualize_data.py -c configs/waters.yaml

# %% import installed packages
import sys
import yaml
import getopt

import numpy as np
import matplotlib.pyplot as plt

#import my modules
from dataset import SegDataset

# %% parameters

# default configuration file path
# configuration file specifies the dataset path and hyperparameters 
CFG_PATH = 'configs/waters.yaml'


# parse arguments (configuration filepath) from command line
opt, args = getopt.getopt(sys.argv[1:], ["c"], ["cfg="])
print(opt)
for op, value in opt:
    # config file 
    if op in ("-c", "--cfg"):
        CFG_PATH = value
        
print('Loading parameters from configuration file : ', CFG_PATH)        
with open(CFG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
print('Configs:')
print(cfg)

#sys.exit(0)

# %% prepare datasets
# init train, val, test sets
print('Prepare data ...')
W, H = cfg['mimgsize'], cfg['mimgsize']
train_dataset = SegDataset(cfg['root_dir'], "train", 
                           n_classes=1, imgH=H, imgW=W, apply_aug = False)
valid_dataset = SegDataset(cfg['root_dir'], "val", 
                           n_classes=1, imgH=H, imgW=W, apply_aug = False)
test_dataset  = SegDataset(cfg['root_dir'], "test", 
                           n_classes=1, imgH=H, imgW=W, apply_aug = False)

# It is a good practice to check datasets don't intersects with each other
train_imgs = train_dataset.get_image_filepaths()
val_imgs = valid_dataset.get_image_filepaths()
test_imgs = test_dataset.get_image_filepaths()
assert set(test_imgs).isdisjoint(set(train_imgs))
assert set(test_imgs).isdisjoint(set(val_imgs))
assert set(val_imgs).isdisjoint(set(train_imgs))
print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")


#------------------------------------------------------------------------------
# lets look at some randomly selected samples
num = 5
ds = train_dataset

idxs = np.int32(np.random.random(num)*(len(ds)-1))
for i in idxs:
    sample = ds[i]
    plt.subplot(1,2,1)
    plt.imshow(sample["oimage"]) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["omask"])  # for visualization we have to remove 3rd dimension of mask
    plt.show()

