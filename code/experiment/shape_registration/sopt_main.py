#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:27:49 2022

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
item='castle'
exp_num='castle'

from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   
data_path=parent_path+'/experiment/shape_registration/data/test2'

data=torch.load(data_path+'/data'+item+'.pt')
label_L=['0','1','2','3']
L=['10k','5k','8k','9k']
label='2'
n_point='/8k'

X0=data['X0'].to(torch.float32)
Y0=data['Y0'+label].to(torch.float32)
X1=data['X1'].to(torch.float32)
Y1=data['Y1'+label].to(torch.float32)
param=data['param']

rotation_op=param['rotation_op']
scalar_op=param['scalar_op']
beta_op=param['beta_op']
print('rotation_op',rotation_op)
#print('N_noise',N_noise)
N=Y0.shape[0]

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
ax.set_xlim3d(-40,15)
ax.set_ylim3d(-5,20)
ax.set_zlim3d(-20,20)
plt.legend(loc='upper right')
plt.savefig('experiment/shape_registration/result/'+exp_num+n_point+'/sopt/init'+label+'.jpg')
plt.show()
plt.close()


# initlize 
rotation=torch.eye(3,dtype=torch.float32)
scalar=1.0 #

beta=0.0 #torch.mean(X1,0)-torch.mean(scalar*Y1@rotation,0)

paramlist=[]
#n_projections=1
Lambda=np.float32(150)
Delta=Lambda*0.1
X1_hat=Y1T@rotation*scalar+beta   

# A=sopt(X1_hat,X1T,Lambda,n_iteration,'orth')
A=sopt_correspondence(X1_hat,X1T,Lambda,100,'orth')
mass_diff=0
for epoch in range(n_iteration):
    # A.get_directions()
    # A.get_one_projection(epoch)
    # A.get_plans()
    # loss,mass=A.sliced_cost()
    # mass_diff=mass.item()-N
    # n=A.X_take.shape[0]
    
    #A.X[A.Lx]+=(A.Y_take-A.X_take).reshape((n,1))*A.projections[epoch]
    #extract the paired data
    #Y1_take=Y1T[A.Lx]
    #X1_hat_take=A.X[A.Lx]
    
    A.get_directions()
    A.correspond(N)
    Y1_take=Y1T[A.Lx]
    X1_hat_take=A.X[A.Lx]
    
    rotation,scalar=recover_rotation(X1_hat_take,Y1_take)
    scalar=torch.sqrt(torch.trace(torch.cov(X1_hat_take.T))/torch.trace(torch.cov(Y1_take.T)))
    beta=torch.mean(X1_hat_take,0)-torch.mean(scalar*Y1_take@rotation,0)
    A.X=Y1T@rotation*scalar+beta
    
    
    if mass_diff>N*0.012:
        A.Lambda-=Delta
      
    if mass_diff<-N*0.003:
        A.Lambda+=Delta
    if A.Lambda<=Delta:
        A.Lambda=Delta
        Delta=Delta/2

    

    param={}
    param['rotation']=rotation
    param['beta']=beta
    param['scalar']=scalar
    paramlist.append(param)

    
    # 
    
    if epoch%20==0 or epoch<=50 or epoch==n_iteration-1:
      print('training Epoch {}/{}'.format(epoch, n_iteration))
      print('lambda',A.Lambda)
      print('mass_diff',mass_diff)
      print('scalar',scalar)
      print('rotation',rotation)
      print('beta',beta)
    

      X1_hat_c=A.X.clone().detach().cpu()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(X1_hat_c[:,0],X1_hat_c[:,1],X1_hat_c[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
      ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=0.3,label='target') # plot the point (2,3,4) on the figure
      ax.set_xlim3d(-40,15)
      ax.set_ylim3d(-5,20)
      ax.set_zlim3d(-20,20)
      plt.legend(loc='upper right')
      plt.savefig('experiment/shape_registration/result/'+exp_num+n_point+'/sopt/'+str(epoch)+'.jpg')
      plt.show()
      plt.close()
      
    
      #print('loss is ',loss.item())
      print('-' * 10)

torch.save(paramlist,'experiment/shape_registration/result/'+exp_num+n_point+'/sopt_param.pt')
