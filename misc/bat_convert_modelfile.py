# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:36:28 2024

@author: renxi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:06:18 2022

@author: renxi
"""

## split test data 

import os
import torch
import csv
import pandas

###################################################################
ROOT_DIR = 'D:\\GeoData\\DLData\\vehicle_seg\\out\\100epochs'

files = os.listdir(ROOT_DIR)

for f in files:
    f = f.lower()
    if not f.endswith('pt'):
        continue
    fp = os.path.join(ROOT_DIR, f)
    print('\n')
    print(fp)    
    m = torch.load(fp)
    print(m.keys())
    
    bname, ext = os.path.splitext(f)
    logf = bname[0:-5]+'_log.csv'
    logfp = os.path.join(ROOT_DIR, logf)
    if os.path.exists(logfp):
        df = pandas.read_csv(logfp)
        #get val_score values
        scores = df.values[:,3]
        best_score = scores.max()
        last_score = scores[-1]
        if bname[-5:].lower()=='_best':
            epoch = scores.argmax()+1
            print('best score:%f  epochs:%d' % (best_score, epoch))
        else:
            epoch = scores.shape[0]
            print('last score:%f  epochs:%d' % (last_score, epoch))
    else:
        print('No training information')
        epoch = 100
        sc_val = 0.5
    
    #if 'lr' not in m.keys():
    m['lr'] = 0.0001
    #if 'epochs' not in m.keys():
    m['epochs'] = epoch
    #if 'score' not in m.keys():
    m['best_score'] = best_score        
    m['last_score'] = last_score        
    
    print(m.keys())
    
    torch.save(m, fp)
        
        
    