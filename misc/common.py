# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:33:19 2024

@author: renxi
"""
import numpy as np


def log_csv(train_loss, val_loss, train_score, val_score, fpath):
    with open(fpath, 'w') as fo:
        print('train_loss, val_loss, train_score, val_score', file=fo)
        for l1, l2, s1, s2 in zip(train_loss, val_loss, train_score, val_score):
            print('%.3f, %.3f, %3f, %3f' % (l1,l2,s1,s2), file=fo)
            
            