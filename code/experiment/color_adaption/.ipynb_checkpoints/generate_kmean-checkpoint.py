#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 19:49:56 2022

@author: baly
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread,imsave
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
from skimage.segmentation import slic
import os
import sys
import ot
import time
work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


exp_num='all'
data_path='experiment/color_adaption/data/'+exp_num
data_path='experiment/color_adaption/data/'+exp_num
num_list=[2,3]
for number in num_list:
    I1=imread(data_path+'/source'+str(number)+'.jpg').astype(np.float64) / 256
    I2=imread(data_path+'/target'+str(number)+'.jpg').astype(np.float64) / 256
    M1,N1,C=I1.shape
    M2,N2,C=I2.shape
    X1=I1.reshape(-1,C)
    X2=I2.reshape(-1,C)

    N1=5000
    kmeans_S=KMeans(n_clusters=N1).fit(X1)
    torch.save(kmeans_S,data_path+'/kmeans_S'+str(number)+'_'+str(N1)+'.pt')
    N2=N1*2
    kmeans_T=KMeans(n_clusters=N2).fit(X2)
    torch.save(kmeans_T,data_path+'/kmeans_T'+str(number)+'_'+str(N2)+'.pt')