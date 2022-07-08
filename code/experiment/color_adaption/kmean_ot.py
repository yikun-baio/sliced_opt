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
import ot
import sys
work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


from sopt2.library import *
from sopt2.sliced_opt import *   
from torch import optim





task_num='1'
data_path='experiment/color_adaption/data/task'+task_num
I1=imread(data_path+'/source.jpg')
I2=imread(data_path+'/target.jpg')

n_clusters=2000
M1,N1,C=I1.shape
M2,N2,C=I2.shape
data1=I1.reshape(-1,C)
data2=I2.reshape(-1,C)

kmeans=torch.load('experiment/color_adaption/data/task'+task_num+'/kmean'+str(n_clusters)+'.pt')
kmean1=kmeans['source']
kmean2=kmeans['target']

label1=kmean1.predict(data1)
centroid1=kmean1.cluster_centers_
centroid2=kmean2.cluster_centers_
centroid1=centroid1.astype(np.float32)
centroid2=centroid2.astype(np.float32)
device='cpu'
dtype=torch.float

X1=torch.tensor(centroid1,device=device,requires_grad=True,dtype=dtype)
X2=torch.tensor(centroid2,device=device,dtype=dtype)
error2_list=[]
nb_iter_max=400
n_projections=C*15
optimizer=optim.Adam([X1],lr=0.1,weight_decay=0)
# show the Wasserstein distance 
mu=np.ones(n_clusters)
nu=np.ones(n_clusters)

for epoch in range(0,nb_iter_max):
    optimizer.zero_grad()
    cost_M=cost_matrix_T(X1,X2)
    cost_M1=cost_M.detach().cpu().numpy()
    plan=ot.lp.emd(mu,nu,cost_M1)
    plan_T=torch.from_numpy(plan).to(device)
    loss=torch.sum(cost_M*plan_T)
    loss.backward()


    if epoch%10==0:
        print('training Epoch {}/{}'.format(epoch, nb_iter_max))
        print('gradient is',torch.norm(X1.grad).item())
        print('loss is ',loss.item())
        print('-' * 10)
    optimizer.step()

    grad_norm=torch.norm(X1.grad)
    if grad_norm>=20:
        optimizer.param_groups[0]['lr']=2
    elif grad_norm>=10:
        optimizer.param_groups[0]['lr']=1
    elif grad_norm>=5:
        optimizer.param_groups[0]['lr']=1
    elif grad_norm>=1:
        optimizer.param_groups[0]['lr']=0.5
    else:
        break

    


centroid1_f=X1.detach().numpy()
I1_f=centroid1_f[label1,:].reshape(M1,N1,C)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(I1.astype(np.uint8))
ax[1].imshow(I1_f.astype(np.uint8))
plt.show()
torch.save(centroid1_f,'experiment/color_adaption/results/ot.pt')

imsave('experiment/color_adaption/results/ot.jpg',I1_f.astype(np.uint8))

  
