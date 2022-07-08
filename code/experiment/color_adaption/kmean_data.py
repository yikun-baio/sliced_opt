#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:14:48 2022

@author: baly
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import os
import sys
import torch
work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
task_num='1'
data_path='experiment/color_adaption/data/task'+task_num
I1=imread(data_path+'/source.jpg')
I2=imread(data_path+'/target.jpg')

n_clusters=2000

M1,N1,C=I1.shape
data1=I1.reshape(-1,C)
kmean1=KMeans(n_clusters=n_clusters).fit(data1)
label1=kmean1.predict(data1)
centroid1=kmean1.cluster_centers_
I1_es=centroid1[label1,:].reshape(M1,N1,C)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(I1.astype(np.uint8))
ax[1].imshow(I1_es.astype(np.uint8))


M2,N2,C=I2.shape
data2=I2.reshape(-1,C)
kmean2=KMeans(n_clusters=n_clusters).fit(data2)
label2=kmean2.predict(data2)
centroid2=kmean2.cluster_centers_
I2_es=centroid2[label2,:].reshape(M2,N2,C)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(I2.astype(np.uint8))
ax[1].imshow(I2_es.astype(np.uint8))


kmeans={}
kmeans['source']=kmean1
kmeans['target']=kmean2
torch.save(kmeans,data_path+'/kmean'+str(n_clusters)+'.pt')



# ## color adation: 

# from library import *
# from sliced_opt import *    

# device = "cuda" if torch.cuda.is_available() else "cpu"
# #device='cpu'
# k_centroids_source_torch=torch.tensor(k_centroids_source,device=device,requires_grad=True,dtype=torch.half)
# k_centroids_target_torch=torch.tensor(k_centroids_target,device=device,dtype=torch.half)
# mu=torch.ones([n_clusters])
# nu=torch.ones([n_clusters])

# nb_iter_max=200
# k_centroids_all = np.zeros((nb_iter_max, k_centroids_source_torch.shape[0], C))

# error1_list=[]

# #method 1, kmean+OT
# for epoch in range(nb_iter_max):
#     M=cost_matrix(k_centroids_source_torch,k_centroids_target_torch)
#     plan = ot.lp.emd(mu, nu,M)
#     loss=sum(sum(M*plan))
#     loss.backward()
#     print('training Epoch {}/{}'.format(epoch+1, nb_iter_max))
#     print('gradient is',torch.norm(k_centroids_source_torch.grad))
#     print('-' * 10)
#     with torch.no_grad():
#         lr1 =5e-2
#         #print(grad.item())
#         k_centroids_source_torch-=k_centroids_source_torch.grad*lr1
#         k_centroids_source_torch.grad.zero_()
#         error1=abs(loss.clone().detach().cpu().numpy())
#         error1_list.append(error1)
#         k_centroids_all[epoch, :, :] = k_centroids_source_torch.clone().detach().cpu().numpy()

# k_centroids_source_final=k_centroids_source_torch.clone().detach().cpu().numpy()
# Image_source_estimate_final=k_centroids_source_final[label_source,:].reshape(M1,N1,C)
# fig,ax=plt.subplots(1,2,figsize=(10,5))
# ax[0].imshow(Image_source.astype(np.uint8))
# ax[1].imshow(Image_source_estimate_final.astype(np.uint8))






# 

