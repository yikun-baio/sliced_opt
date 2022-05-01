# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:10:10 2022

@author: laoba
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
import ot
import os

# load picture and reduce the resolution 
#current_path = os.path.dirname(os.path.realpath(__file__))
local_path_head='G:/My Drive'
parent_path='/Github/Yikun-Bai/Sliced-Optimal-Partial-Transport/Code'
parent_path=local_path_head+parent_path
dataset_path =parent_path+'/datasets'
data_path=parent_path+'/experiment2/Color_adaption'
Image_path = dataset_path+'/Images/task1'

lab_path=parent_path+'/sopt'
Image_path = dataset_path+'/Images/task1'

os.chdir(lab_path)

n_clusters=20
Image_source=imread(Image_path+'/source.jpg')
Image_target=imread(Image_path+'/target.jpg')

M1,N1,C=Image_source.shape
data_source=Image_source.reshape(-1,C)
kmeans_source=KMeans(n_clusters=n_clusters).fit(data_source)
label_source=kmeans_source.predict(data_source)

k_centroids_source=kmeans_source.cluster_centers_
Image_source_estimate=k_centroids_source[label_source,:].reshape(M1,N1,C)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(Image_source.astype(np.uint8))
ax[1].imshow(Image_source_estimate.astype(np.uint8))


M2,N2,C=Image_target.shape
data_target=Image_target.reshape(-1,C)
kmeans_target=KMeans(n_clusters=n_clusters).fit(data_target)
label_target=kmeans_target.predict(data_target)

k_centroids_target=kmeans_target.cluster_centers_
Image_target_estimate=k_centroids_target[label_target,:].reshape(M2,N2,C)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(Image_target.astype(np.uint8))
ax[1].imshow(Image_target_estimate.astype(np.uint8))


kmeans={}
kmeans['source']=kmeans_source
kmeans['target']=kmeans_target



## color adation: 

from library import *
from sliced_opt import *    

device = "cuda" if torch.cuda.is_available() else "cpu"
#device='cpu'
k_centroids_source_torch=torch.tensor(k_centroids_source,device=device,requires_grad=True,dtype=torch.half)
k_centroids_target_torch=torch.tensor(k_centroids_target,device=device,dtype=torch.half)
mu=torch.ones([n_clusters])
nu=torch.ones([n_clusters])

nb_iter_max=200
k_centroids_all = np.zeros((nb_iter_max, k_centroids_source_torch.shape[0], C))

error1_list=[]

#method 1, kmean+OT
for epoch in range(nb_iter_max):
    M=cost_matrix(k_centroids_source_torch,k_centroids_target_torch)
    plan = ot.lp.emd(mu, nu,M)
    loss=sum(sum(M*plan))
    loss.backward()
    print('training Epoch {}/{}'.format(epoch+1, nb_iter_max))
    print('gradient is',torch.norm(k_centroids_source_torch.grad))
    print('-' * 10)
    with torch.no_grad():
        lr1 =5e-2
        #print(grad.item())
        k_centroids_source_torch-=k_centroids_source_torch.grad*lr1
        k_centroids_source_torch.grad.zero_()
        error1=abs(loss.clone().detach().cpu().numpy())
        error1_list.append(error1)
        k_centroids_all[epoch, :, :] = k_centroids_source_torch.clone().detach().cpu().numpy()

k_centroids_source_final=k_centroids_source_torch.clone().detach().cpu().numpy()
Image_source_estimate_final=k_centroids_source_final[label_source,:].reshape(M1,N1,C)
fig,ax=plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(Image_source.astype(np.uint8))
ax[1].imshow(Image_source_estimate_final.astype(np.uint8))






# 