#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 21:24:42 2022

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
work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


from sopt2.library import *
from sopt2.sliced_opt import *   
from sopt2.lib_color import *   
from torch import optim

work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import ot


rng = np.random.RandomState(42)



exp_num='1'
data_path='experiment/color_adaption/data/'+exp_num
I1=imread(data_path+'/source.jpg').astype(np.float64) / 256
I2=imread(data_path+'/target.jpg').astype(np.float64) / 256

M1,N1,C=I1.shape
M2,N2,C=I2.shape
X1=I1.reshape(-1,C)
X2=I2.reshape(-1,C)
rng = np.random.RandomState(42)
nb1 = 500
nb2 = 1000
idx1 = rng.randint(X1.shape[0], size=(nb1,))
idx2 = rng.randint(X2.shape[0], size=(nb2,))
Xs = X1[idx1, :]
Xt = X2[idx2, :]
start_time=time.time()
XsT=torch.from_numpy(Xs).to(dtype=torch.float)
XtT=torch.from_numpy(Xt).to(dtype=torch.float)
L=[]
n_projections=600
Lambda=10.0
Lambda_list=torch.full((n_projections,),Lambda)

A=spot(XsT.clone(),XtT.clone(),n_projections)
A.n_projections=n_projections
A.get_directions()
A.correspond()
transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))
end_time=time.time()
wall_time=end_time-start_time
I1t = minmax(mat2im(transp_Xs, I1.shape))
plt.figure()
plt.axis('off')
#    plt.title('Image Source')
plt.imshow(I1t)
plt.savefig('experiment/color_adaption/results/'+exp_num+'/spot.png',format='png',dpi=2000)
plt.close()
result={}
result['transp_Xs']=transp_Xs
result['wall_time']=wall_time
torch.save(wall_time,'experiment/color_adaption/results/'+exp_num+'/spot'+'.pt')

