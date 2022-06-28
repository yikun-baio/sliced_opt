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
from sopt2.lib_shape import *



split = 'train'
dataset_name ='modelnet40'
root = parent_path+'/experiment/shape_registration/data'
#save_root = os.path.join(root+'/testdata', dataset_name)
save_root=root+'/test'
if not os.path.exists(save_root):
    os.makedirs(save_root)

from experiment.shape_registration.dataset import Dataset


    
item=120

d = Dataset(root=root, dataset_name=dataset_name,num_points=2048,   split=split, 
        random_rotate=False, load_name=True)



#for item in range(d.__len__()):
print("datasize:", d.__len__())
pts, lb, label,device = d[item]
X=pts.clone()
print('item',item)
print('label',label)


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],s=2.5) # plot the point   (2,3,4) on the figure
plt.show()
theta=torch.tensor([-torch.pi/3,torch.pi/3,-3/4*torch.pi])
rotation=rotation_3d_2(theta,'in')
scalar=0.5 #0.6
beta=torch.tensor([1.6,0.8,-0.3]) #torch.tensor([1.8,0.5,0.5])
Y=scalar*X@rotation+beta
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],s=0.3,label='target') # plot the point (2,3,4) on the figure
ax.scatter(Y[:,0],Y[:,1],Y[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
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
# test if the data is symmetric 
recover_rotation(X,Y)

torch.save(data,save_root+'/data'+str(item)+'.pt')

#X1T_c=X-torch.mean(X,0)
#Y1T_c=Y-torch.mean(Y,0)
#U1,S1,V1=torch.pca_lowrank(X1T_c)
#U2,S2,V2=torch.pca_lowrank(Y1T_c)



    
    
    
    
