#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 20:03:10 2022

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



exp_num='1'
data_path='experiment/color_adaption/data/'+exp_num
data_path='experiment/color_adaption/data/'+exp_num
I1=imread(data_path+'/source.jpg').astype(np.float64) / 256
I2=imread(data_path+'/target.jpg').astype(np.float64) / 256

M1,N1,C=I1.shape
M2,N2,C=I2.shape
X1=I1.reshape(-1,C)
X2=I2.reshape(-1,C)

N1=5000
N2=5000
kmeans_X1=KMeans(n_clusters=N1).fit(X1)
kmeans_X2=KMeans(n_clusters=N2).fit(X2)
kmeans={}
kmeans['source']=kmeans_X1
kmeans['target']=kmeans_X2
torch.save(kmeans,data_path+'/kmean'+str(N1)+'-'+str(N2)+'.pt')

rng = np.random.RandomState(42)
nb1 = 2000
nb2 = 2000
idx1 = rng.randint(X1.shape[0], size=(nb1,))
idx2 = rng.randint(X2.shape[0], size=(nb2,))

Xs = X1[idx1, :]
Xt = X2[idx2, :]

#XsT=torch.from_numpy(Xs).to(dtype=torch.float)
#XtT=torch.from_numpy(Xt).to(dtype=torch.float)


# EMDTransport
ot_emd = ot.da.EMDTransport()
ot_emd.fit(Xs=Xs, Xt=Xt)

# SinkhornTransport
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
# prediction between images (using out of sample prediction as in [6])
transp_Xs_emd = ot_emd.transform(Xs=X1)
#transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)

transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
#transp_Xt_sinkhorn = ot_sinkhorn.inverse_transform(Xt=X2)

I1t = minmax(mat2im(transp_Xs_emd, I1.shape))
#I2t = minmax(mat2im(transp_Xt_emd, I2.shape))

I1te = minmax(mat2im(transp_Xs_sinkhorn, I1.shape))
#I2te = minmax(mat2im(transp_Xt_sinkhorn, I2.shape))

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(I1t)
plt.savefig('experiment/color_adaption/results/'+exp_num+'/ot.jpg',format="png",dpi=2000)
plt.close()
torch.save(transp_Xs_emd,'experiment/color_adaption/results/'+exp_num+'/ot'+'.pt')

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(I1te)
plt.savefig('experiment/color_adaption/results/'+exp_num+'/ot_sinkhorn.jpg')
plt.close()
torch.save(transp_Xs_sinkhorn,'experiment/color_adaption/results/'+exp_num+'/ot_sinkhorn'+'.pt')


