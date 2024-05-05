# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:14:49 2023

@author: renxi
"""

import os, sys
import pandas as pd
import matplotlib.pylab as plt


def plot_csv(root_dir, filename):
    filepath = os.path.join(root_dir, filename)
    if not os.path.exists(filepath):
        print('File does not exist: ', filepath)
        sys.exit(0)
        
    data = pd.read_csv(filepath)
    print(data.columns)
    
    
    def plot_img(dataList, nameList, ylabel= 'loss', title='Plot', save_path='Plot.png'):
        plt.style.use("ggplot")
        plt.figure()
        for ds, dname in zip(dataList, nameList): 
            plt.plot(ds, label=dname)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")    
        plt.savefig(save_path, dpi=200)
    
    
    #plot F1-score
    f1_train = data['train_f1-score'].dropna()
    f1_val   = data['valid_f1-score'].dropna()
    print(f1_train)
    print(f1_val)
    plot_img([f1_train, f1_val], ['train_F1', 'Validation_F1'],
             ylabel='F1-score', title='train-validation F1-score',
             save_path = os.path.join(root_dir, 'fig_f1-score.png'))
    
    
    #plot Precision
    pre_train = data['train_precision'].dropna()
    pre_val   = data['valid_precision'].dropna()
    print(pre_train)
    print(pre_val)
    plot_img([pre_train, pre_val], ['train_prec.', 'Validation_prec.'],
             ylabel='Precision', title='train-validation precision',
             save_path = os.path.join(root_dir, 'fig_prec.png'))
    
    #plot recall
    rec_train = data['train_recall'].dropna()
    rec_val   = data['valid_recall'].dropna()
    print(rec_train)
    print(rec_val)
    plot_img([rec_train, rec_val], ['train_recall', 'Validation_recall'],
             ylabel='Recall', title='train-validation recall',
             save_path = os.path.join(root_dir, 'fig_recall.png'))         
    
    #plot IOU
    iou_train = data['train_dataset_iou'].dropna()
    iou_val   = data['valid_dataset_iou'].dropna()
    print(iou_train)
    print(iou_val)
    plot_img([iou_train, iou_val], ['train_IOU', 'Validation_IOU'],
             ylabel='iou-score', title='train-validation IOU',
             save_path = os.path.join(root_dir, 'fig_iou.png'))


if __name__ == '__main__':
    root_dir = 'D:/GeoData/DLData/Oxford_iiit_pet/logs/UNET_resnet34/version_1'
    root_dir = 'D:/GeoData/DLData/Buildings/logs/UnetPlusPlus_resnet34/version_6'
    filename = 'metrics.csv'
    plot_csv(root_dir, filename)
    




