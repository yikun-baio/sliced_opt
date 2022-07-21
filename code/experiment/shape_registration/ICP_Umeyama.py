#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:37:01 2022

@author: baly
"""

import sys
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import ot

work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)

# load the data
label='30'
exp_num='1'
from sopt2.library import *
from sopt2.lib_shape import *
data_path=parent_path+'/experiment/shape_registration/data/test'

data=torch.load(data_path+'/data_noise'+label+'.pt')
L=[30,60,39,105,340]
X0=data['X0']
Y0=data['Y0']
X1=data['X1']
Y1=data['Y1']
N=data['N']
N_noise=data['N_noise']
param=data['param']
theta_op=param['theta_op']
scalar_op=param['scalar_op']
beta_op=param['beta_op']
print('theta_op',theta_op)
print('N_noise',N_noise)
N_a=206
X1=X1[0:N+N_noise-N_a].clone()
Y1=Y1[0:N+N_noise-N_a].clone()
device='cpu'
dtype=torch.float32
X0T=X0.to(device).clone()
Y0T=Y0.to(device).clone()
X1T=X1.to(device).clone()
Y1T=Y1.to(device).clone()
n_iteration=300

# compute the parameter error
print('original figure')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=0.3,label='target') # plot the point (2,3,4) 
ax.set_xlim3d(-1,2.5)
ax.set_ylim3d(-1.5,1.5)
ax.set_zlim3d(-1,1)
plt.legend(loc='upper right')
plt.savefig('experiment/shape_registration/result/exp'+exp_num+'/icp_Umeyama/init'+label+'.jpg')
plt.show()
plt.close()


# initlize 
rotation=torch.eye(3).to(dtype=torch.float32)

X_c=X1-torch.mean(X1,0)
Y_c=Y1-torch.mean(Y1,0)
scalar=torch.sqrt(torch.trace(X_c.T@X_c)/torch.trace(Y_c.T@Y_c))

beta=torch.mean(X1,0)-torch.mean(scalar*Y1@rotation,0)

paramlist=[]
#n_projections=1
Lambda=np.float32(0.08)
Delta=Lambda*0.1


for epoch in range(n_iteration):
    X1_hat=Y1T@rotation*scalar+beta
    M=cost_matrix_T(X1_hat,X1)
    n,m=M.shape 
    argmin_X1=M.argmin(dim=1)
    X1_take=X1[argmin_X1]
    X1_hat_take=X1_take
    rotation,scalar=recover_rotation(X1_hat_take,Y1)
    beta=torch.mean(X1_hat_take,0)-torch.mean(scalar*Y1@rotation,0)
    
    if epoch%30==0 or epoch<=50 or epoch==n_iteration-1:
      print('training Epoch {}/{}'.format(epoch, n_iteration))
      print('scalar',scalar)
      print('rotation',rotation)
      print('beta',beta)
    

      X1_hat_c=X1_hat_take.clone().detach().cpu()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(X1_hat_c[:,0],X1_hat_c[:,1],X1_hat_c[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
      ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=0.3,label='target') # plot the point (2,3,4) on the figure
      ax.set_xlim3d(-1,2.5)
      ax.set_ylim3d(-1.5,1.5)
      ax.set_zlim3d(-1,1)
      plt.legend(loc='upper right')
      plt.savefig('experiment/shape_registration/result/exp'+exp_num+'/icp_Umeyama/'+str(epoch)+'.jpg')
      plt.show()
      plt.close()
      
    
      print('-' * 10)

torch.save(paramlist,'experiment/shape_registration/result/exp'+exp_num+'/icp_Umeyama_param.pt')
    
    
    
    

 