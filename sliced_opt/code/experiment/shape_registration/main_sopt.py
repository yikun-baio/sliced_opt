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

work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
label='120'
exp_num='3'
from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   
os.getcwd()
data_path=parent_path+'/experiment/shape_registration/data/test'

data=torch.load(data_path+'/data_noise'+label+'.pt')

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
N_a=0
X1=X1[0:N+N_noise-N_a].clone()
Y1=Y1[0:N+N_noise-N_a].clone()
device='cpu'
dtype=torch.float32
X0T=X0.to(device)
Y0T=Y0.to(device)
X1T=X1.to(device)
Y1T=Y1.to(device)
n_iteration=300


# compute the parameter error
print('original figure')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=0.3,label='target') # plot the point (2,3,4) on the figure
ax.set_xlim3d(-1,2.5)
ax.set_ylim3d(-1.5,1.5)
ax.set_zlim3d(-1,1)
plt.legend(loc='upper right')
plt.savefig('experiment/shape_registration/result/exp'+exp_num+'/sopt/init.jpg')
plt.show()
plt.close()


# initlize 
scalar=torch.tensor(1,dtype=dtype,requires_grad=True,device=device)

#initilize theta
theta=init_angle(X1,Y1)
theta=theta.to(device).requires_grad_()
print('original theta is ',theta)

# initilize beta 
rotation=rotation_3d_2(theta,'re')
X1_hat=Y1T@rotation*scalar
mean_X1_hat=torch.mean(X1_hat.clone().detach(),0)*(N+N_noise-N_a)/N
mean_X1=torch.mean(X1.clone(),0)*(N+N_noise-N_a)/N

beta=mean_X1-mean_X1_hat
beta=beta.to(device).requires_grad_()



#beta=torch.tensor([0,0,0],dtype=dtype,requires_grad=True)
optimizer1=optim.Adam([scalar],lr=0.1,weight_decay=0.01)
optimizer2=optim.Adam([theta],lr=0.1,weight_decay=0.01)
optimizer3=optim.Adam([beta],lr=0.1,weight_decay=0.01)


# show the Wasserstein distance 
paramlist=[]
n_projections=72
Lambda=np.float32(0.1)
Delta=Lambda*0.1

      

for epoch in range(n_iteration):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    rotation=rotation_3d_2(theta,'re')
    X1_hat=Y1T@rotation*scalar+beta
#    mean_X1=torch.mean(X1,0)*(N+N_noise-N_a)/N
#    mean_X1_hat=torch.mean(X1_hat,0)*(N+N_noise-N_a)/N
#    beta=mean_X1-mean_X1_hat
    X1_hat2=X1_hat #+beta
#    A=sopt_majority(X1_hat2,X1T,Lambda,n_projections,'orth',n_destroy=N_noise-N_a)
    A=sopt_for(X1_hat2,X1T,Lambda,n_projections,'orth')

    loss,mass=A.sliced_cost()
    loss.backward()
    mass_diff=mass.item()-N
    
    if mass_diff>N*0.012:
        Lambda-=Delta
      
    if mass_diff<-N*0.003:
        Lambda+=Delta

    
    if Lambda<=abs(Delta):
        Lambda=abs(Delta)
        Delta=Delta/2
    

    param={}
    param['theta']=theta
    param['beta']=beta
    param['scalar']=scalar
    paramlist.append(param)
    
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    
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
      ax.set_xlim3d(-1,2.5)
      ax.set_ylim3d(-1.5,1.5)
      ax.set_zlim3d(-1,1)
      plt.legend(loc='upper right')
      plt.savefig('experiment/shape_registration/result/exp'+exp_num+'/sopt/'+str(epoch)+'.jpg')
      plt.show()
      plt.close()
      
    
      print('training Epoch {}/{}'.format(epoch, n_iteration))
      print('grad of scalar is',torch.norm(scalar.grad))
      print('grad of theta is',torch.norm(theta.grad))
      #    print('grad of beta is',torch.norm(beta.grad))
      print('loss is ',loss.item())
      print('-' * 10)

torch.save(paramlist,'experiment/shape_registration/result/exp'+exp_num+'/sopt_param.pt')

