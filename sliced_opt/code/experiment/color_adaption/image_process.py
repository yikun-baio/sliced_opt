#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:37:58 2022

@author: baly
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2 
from skimage.io import imread,imsave
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
from skimage.segmentation import slic
import os
import ot
import sys
work_path=os.path.dirname(__file__)

loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
label='sopt1'
n_clusters=1000

# load the data 

task_num='1'
data_path='experiment/color_adaption/data/task'+task_num
I1=imread(data_path+'/source.jpg')
I2=imread(data_path+'/target.jpg')
M1,N1,C=I1.shape
M2,N2,C=I2.shape
data1=I1.reshape(-1,C)
data2=I2.reshape(-1,C)
kmeans=torch.load('experiment/color_adaption/data/task'+task_num+'/kmean'+str(n_clusters)+'.pt')
kmean1=kmeans['source']
kmean2=kmeans['target']
label1=kmean1.predict(data1)

# load the result 
result_path='experiment/color_adaption/results/'
centroid1_f=torch.load(result_path+label+'.pt')

I1_f=centroid1_f[label1,:].reshape(M1,N1,C)
imsave('experiment/color_adaption/results/'+label+'.jpg',I1_f.astype(np.uint8))

delta=I1_f-I1
bilateral = cv2.bilateralFilter(delta, 15, 60, 60)

fig,ax=plt.subplots(1,2,figsize=(15,5))
ax[0].matshow(delta)
ax[1].matshow(bilateral)
plt.show()

# plt.matshow(delta)
# plt.show()
# plt.matshow(bilateral)
# plt.show()


I1_f_bilateral=I1+bilateral 
fig,ax=plt.subplots(1,2,figsize=(15,5))
ax[0].imshow(I1_f.astype(np.uint16))
ax[1].imshow(I1_f_bilateral.astype(np.uint16))
plt.show()
imsave('experiment/color_adaption/results/'+label+'bil.jpg',I1_f_bilateral.astype(np.uint8))

# Save the output.
#cv2.imwrite('taj_bilateral.jpg', bilateral)