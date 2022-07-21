#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:58:45 2022

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
import  time


# choose the data 
label='castle'


work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.lib_shape import *

data_path=parent_path+'/experiment/shape_registration/data/test2'
data=torch.load(data_path+'/data'+str(label)+'.pt')
dtype=torch.float32
theta=data['theta']
X=data['X0']
Y=data['Y0']
beta=data['beta']
scalar=data['scalar']

# compute the optimal parameters 
theta_op=-theta
rotation_op=rotation_3d_2(theta_op,'re')
scalar_op=1/scalar
beta_op=-1/scalar*beta@rotation_op
Y0=Y.clone()
X0=Y0@rotation_op*scalar_op+beta_op
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y0[:,0],Y0[:,1],Y0[:,2],s=0.03,label='source')  
ax.scatter(X0[:,0],X0[:,1],X0[:,2],s=0.03,label='target') 
plt.legend(loc='upper right')
plt.show()



# add noise
per=1/19
N=Y0.shape[0] # of clean data
N_noise=int(per*N) # of noise
nx=40*(torch.rand(N_noise,3)-0.5)#+torch.mean(X0,0)
ny=40*(torch.rand(N_noise,3)-0.5)#+torch.mean(Y0,0)
#nx=nx-torch.mean(nx,0)
#ny=nx-torch.mean(ny,0)
Y1=torch.cat((Y0,ny))
randindex=torch.randperm(N+N_noise)
Y1=Y1[randindex]
X1=torch.cat((X0,nx))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=3,label='source') # plot the point (2,3,4) on the figure
ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=3,label='target') # plot the point 
ax.set_xlim3d(-40,15)
ax.set_ylim3d(-5,20)
ax.set_zlim3d(-20,20)
plt.legend(loc='upper right')
plt.show()
data['X1']=X1 
data['Y1']=Y1 
data['N']=N
data['N_noise']=N_noise
param={}
param['theta_op']=theta_op
param['scalar_op']=scalar_op
param['beta_op']=beta_op

data['param']=param

rotation_es,scalar_es=recover_rotation(X1,Y1)
#scalar_es=torch.sqrt(torch.trace(torch.cov(X0.T))/torch.trace(torch.cov(Y0.T)))
beta_es=torch.mean(X0,0)-torch.mean(scalar_es*Y0@rotation_es,0)
#beta_es=torch.mean(Y0)
torch.save(data,data_path+'/data'+str(label)+'.pt')
