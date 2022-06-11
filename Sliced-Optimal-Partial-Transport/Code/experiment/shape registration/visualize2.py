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

current_path = os.path.dirname(os.path.realpath(__file__))
print(current_path)
start=current_path.find('Sliced-Optimal-Partial-Transport')
parent_path=current_path[0:start+37]
os.chdir(parent_path)
from sopt.library import rotation_matrix_3d



split = 'train'
dataset_name ='modelnet40'
root = parent_path+'/experiment/point_cloud_matching/data'
#save_root = os.path.join(root+'/testdata', dataset_name)
save_root=root+'/test'
if not os.path.exists(save_root):
    os.makedirs(save_root)

from experiment.point_cloud_matching.dataset import Dataset


    

L=[14,20,30,39,44]
label_list=['plant','bottle','airplane','chair','guitar']
theta_list=[torch.tensor([torch.pi/3,-1/3*torch.pi,1/2*torch.pi]),torch.tensor([-torch.pi/2,2/3*torch.pi,torch.pi/3])]
#beta_list=[torch.tensor([1,-1,1]),torch.tensor([-1,1,-1])]
scalar_list=[0.8,1.5]
beta_list=[torch.tensor([1.2,1,0.5],dtype=torch.float32),torch.tensor([-3,3,-3],dtype=torch.float32)]


item=14
 #item =49
d = Dataset(root=root, dataset_name=dataset_name,num_points=2048, split=split, 
            random_rotate=False, load_name=True)


print("datasize:", d.__len__())
pts, lb, label,device = d[item]
X=pts.clone()
#X.dtype=torch.float32
print('label',label)

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=2.5) # plot the point (2,3,4) on the figure
plt.show()
i=0
theta=theta_list[i]
rotation=rotation_matrix_3d(theta)
scalar=scalar_list[i]
beta=beta_list[i]
Y=scalar*pts@rotation+beta
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],s=2.5,label='target') # plot the point (2,3,4) on the figure
ax.scatter(Y[:,0],Y[:,1],Y[:,2],s=2.5,label='source') # plot the point (2,3,4) on the figure
plt.legend()
plt.show()
data={}
data['X']=X
data['Y']=Y
data['theta']=theta
data['beta']=beta
data['scalar']=scalar
data['label']=label
data['item']=item
torch.save(data,save_root+'/data'+str(item))



    
    
    
    
