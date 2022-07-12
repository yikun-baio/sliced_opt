
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:57:37 2022

@author: laoba
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
from torch import optim


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

M1,N1,C=I1.shape
M2,N2,C=I2.shape
data1=I1.reshape(-1,C)
data2=I2.reshape(-1,C)
n_clusters=2000
kmeans=torch.load('experiment/color_adaption/data/task'+task_num+'/kmean'+str(n_clusters)+'.pt')
kmean1=kmeans['source']
kmean2=kmeans['target']

label1=kmean1.predict(data1)
centroid1=kmean1.cluster_centers_.astype(np.float32)
centroid2=kmean2.cluster_centers_.astype(np.float32)
device='cpu'
dtype=torch.float

#N=1500
#Lambda=0.3
X1=torch.tensor(centroid1,device=device,dtype=dtype)
X2=torch.tensor(centroid2,device=device,dtype=dtype)
error2_list=[]
nb_iter_max=3000
n_projections=C*15
Lambda=30
Delta=Lambda*1/10
print('Lambda',Lambda)
A=sopt(X1,X2,Lambda,nb_iter_max,'orth')



for epoch in range(0,nb_iter_max):
    A.get_one_projection(epoch)
    A.get_plans()
    loss,mass=A.sliced_cost()
   # mass_diff=mass.item()-N
    n=A.X_take.shape[0]
    A.X[A.Lx]+=(A.Y_take-A.X_take).reshape((n,1))*A.projections[epoch]
    
    
    # if mass_diff>N*0.012:
    #     A.Lambda-=Delta
    
    # if mass_diff<-N*0.003:
    #     A.Lambda+=Delta
    # if A.Lambda<=Delta:
    #     A.Lambda=Delta
    #     Delta=Delta/2
    if epoch<=50 or epoch%10==0:
        print('mass',mass)
#        print('lambda',A.Lambda)


    
    

centroid1_f=A.X.clone().detach().cpu().numpy()
centroid1_f2=centroid1
cost_M=cost_matrix_d(centroid1_f,centroid2)
Delta=10
n_point=0
for i in range(n_clusters):
    if np.sum(cost_M[i,:]<=Delta)>=1:
        centroid1_f2[i,:]=centroid1_f[i,:]
        n_point+=1

torch.save(centroid1_f,'experiment/color_adaption/results/sopt'+str(Lambda)+'.pt')
torch.save(centroid1_f2,'experiment/color_adaption/results/sopt'+str(Lambda)+'.pt')

print('transfer {} colors'.format(n_point))
I1_f2=centroid1_f2[label1,:].reshape(M1,N1,C)
I1_f=centroid1_f[label1,:].reshape(M1,N1,C)
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(I1.astype(np.uint8))
ax[1].imshow(I1_f.astype(np.uint8))
ax[2].imshow(I1_f2.astype(np.uint8))
plt.show()
imsave('experiment/color_adaption/results/sopt'+str(Lambda)+'.jpg',I1_f.astype(np.uint8))

  
