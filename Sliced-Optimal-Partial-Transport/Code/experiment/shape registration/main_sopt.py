#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:33:29 2022

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

work_path=os.getcwd()
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
sys.path.append(parent_path)
label='20'
from sopt.library import *
from sopt.sliced_opt import *   
os.getcwd()
data_path=parent_path+'/experiment/point_cloud_matching/data/test'

data=torch.load(data_path+'/data_noise'+label+'.pt')

X0=data['X0']
Y0=data['Y0']
X1=data['X1']
Y1=data['Y1']
X1=X1[0:-150]
Y1=Y1[0:-150]
N=data['N']
N_noise=data['N_noise']
param=data['param']
theta_op=param['theta_op']
scalar_op=param['scalar_op']
beta_op=param['beta_op']

device='cpu'
dtype=torch.float32
X0T=X0.to(device)
Y0T=Y0.to(device)
X1T=X1.to(device)
Y1T=Y1.to(device)
n_iteration=200

theta=torch.tensor([0,0,0],dtype=dtype,requires_grad=True,device=device)
scalar=torch.tensor(1,dtype=dtype,requires_grad=True,device=device)

#beta=torch.tensor([0,0,0],dtype=dtype,requires_grad=True)
optimizer=optim.Adam([scalar,theta],lr=0.1,weight_decay=0.01)
#optimizer2=optim.Adam([theta],lr=0.2,weight_decay=0.01)
#optimizer3=optim.Adam([beta],lr=0.1,weight_decay=0.01)

# show the Wasserstein distance 

mu0=1/N*np.ones(N)
nu0=1/N*np.ones(N)
mass=N-2
paramlist=[]
errlist=[]
n_projections=30

Lambda=np.float32(0.2)
Delta=Lambda*0.2


# compute the parameter error
for epoch in range(n_iteration):
    optimizer.zero_grad()
#    optimizer2.zero_grad()
    #  optimizer3.zero_grad()
    rotation=rotation_matrix_3d(theta,'re')
    X1_hat=Y1T@rotation*scalar
    mean_X1=torch.mean(X1T,0)*(N+N_noise)/N
    mean_X1_hat=torch.mean(X1_hat,0)*(N+N_noise)/N
    beta=mean_X1-mean_X1_hat
    
    
    X1_hat2=X1_hat+beta
    A=sopt_for(X1_hat2,X1T,Lambda,n_projections,'orth')
    loss,mass=A.sliced_cost()
    loss.backward()
    mass_diff=mass.item()-N
    
    if mass_diff>N*0.001:
        Lambda-=Delta
      
    if mass_diff<-N*0.004:
        Lambda+=Delta

    
    if Lambda<=abs(Delta):
        Lambda=abs(Delta)
        Delta=Delta/2
    

    param={}
    param['theta']=theta
    param['beta']=beta
    param['scalar']=scalar
    paramlist.append(param)
    
    optimizer.step()
#    optimizer2.step()
    
    # 
    
    if epoch%10==0 or epoch<=50:
      print('lambda',Lambda)
      print('mass_diff',mass_diff)
      print('scalar',scalar)
      print('theta',theta)
      print('beta',beta)
    

      X1_hat_c=X1_hat2.clone().detach().cpu()
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(X1_hat_c[:,0],X1_hat_c[:,1],X1_hat_c[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
      ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=0.3,label='target') # plot the point (2,3,4) on the figure
      ax.set_xlim3d(-1,1.5)
      ax.set_ylim3d(-1.5,1.5)
      ax.set_zlim3d(-1,1)
      plt.legend(loc='upper right')
      plt.savefig('experiment/point_cloud_matching/result/exp1/sopt/'+str(epoch)+'.jpg')
      #plt.show()
      #plt.close()
      
    
      print('training Epoch {}/{}'.format(epoch, n_iteration))
      print('grad of scalar is',torch.norm(scalar.grad))
      print('grad of theta is',torch.norm(theta.grad))
      #    print('grad of beta is',torch.norm(beta.grad))
      print('loss is ',loss.item())
      print('-' * 10)

torch.save(paramlist,'experiment/point_cloud_matching/result/exp1/sopt_param.pt')
torch.save(errlist,'experiment/point_cloud_matching/result/exp1/sopt_err.pt')
