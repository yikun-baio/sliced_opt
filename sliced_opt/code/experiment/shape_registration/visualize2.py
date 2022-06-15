#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:36:23 2022

@author: baly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import sys
work_path=os.path.dirname(__file__)
print('work_path is', work_path)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt.library import rotation_matrix_3d



split = 'train'
dataset_name ='modelnet40'
root = parent_path+'/experiment/shape_registration/data'
#save_root = os.path.join(root+'/testdata', dataset_name)
save_root=root+'/test'
if not os.path.exists(save_root):
    os.makedirs(save_root)

from experiment.shape_registration.dataset import Dataset


    

L_chair=[]
theta_list=[torch.tensor([torch.pi/3,2/5*torch.pi,-1/3*torch.pi])]
#beta_list=[torch.tensor([1,-1,1]),torch.tensor([-1,1,-1])]
scalar_list=[0.8]
beta_list=[torch.tensor([1.8,0.5,0.5],dtype=torch.float32)]

item=2130
#item =49
d = Dataset(root=root, dataset_name=dataset_name,num_points=2048,   split=split, 
        random_rotate=False, load_name=True)



#for item in range(d.__len__()):
print("datasize:", d.__len__())
pts, lb, label,device = d[item]
X=pts.clone()
#X.dtype=torch.float32
print('item',item)
print('label',label)
if label=='chair':
    L_chair.append(item)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=2.5) # plot the point   (2,3,4) on the figure
plt.show()
i=0
theta=theta_list[i]
rotation=rotation_matrix_3d(theta)
scalar=scalar_list[i]
beta=beta_list[i]
Y=scalar*pts@rotation+beta
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],s=0.4,label='target') # plot the point (2,3,4) on the figure
ax.scatter(Y[:,0],Y[:,1],Y[:,2],s=0.4,label='source') # plot the point (2,3,4) on the figure
plt.legend()
plt.show()
plt.close()
data={}
data['X']=X
data['Y']=Y
data['theta']=theta
data['beta']=beta
data['scalar']=scalar
data['label']=label
data['item']=item
torch.save(data,save_root+'/data'+str(item)+'.pt')

X1T_c=X-torch.mean(X,0)
U1,S1,V1=torch.pca_lowrank(X1T_c)
v11=-V1[:,0]
v12=-V1[:,1]
v13=V1[:,2]
#v12=-V1[0,:]

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],s=0.4,label='target') # plot the point (2,3,4) on the figure
ax.scatter(Y[:,0],Y[:,1],Y[:,2],s=0.4,label='source') # plot the point (2,3,4) on the figure
ax.scatter(v11[0],v11[1],v11[2],s=30,label='pca1')
ax.scatter(v12[0],v12[1],v12[2],s=30,label='pca2')
ax.scatter(v13[0],v13[1],v13[2],s=30,label='pca3')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.legend(loc='upper right')
plt.show()

    
    
    
    
