


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:27:49 2022

@author: baly
"""


import sys
import open3d as o3d
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
item='/mumble_sitting' 
#'/stanford_bunny' #'/witchcastle' #'mumble_sitting'
exp_num='/mumble_sitting' 
#'/stanford_bunny'#'/witchcastle' #'mumble_sitting'

from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   
label_L=['0','1','2','3']
L=['10k','9k','8k','7k']
label='0'
n_point='/10k'
per_s='-5p'
data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
save_path='experiment/shape_registration/result'+exp_num+n_point
data=torch.load(data_path+item+'.pt')


X0=data['X0'].to(torch.float32)
Y0=data['Y0'+label].to(torch.float32)
X1=data['X1'+per_s].to(torch.float32)
Y1=data['Y1'+label+per_s].to(torch.float32)
#param=data['param']


N=Y1.shape[0]

device='cpu'
dtype=torch.float32
X0T=X0.to(device).clone()
Y0T=Y0.to(device).clone()
X1T=X1.to(device).clone()
Y1T=Y1.to(device).clone()


print('original figure')
fig = plt.figure(figsize=(10,10))
ncolors = len(plt.rcParams['axes.prop_cycle'])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
plt.axis('off')
ax.set_facecolor("grey")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.legend(loc='upper right',scatterpoints=100)
ax.set_xlim3d(-45,45)
ax.set_ylim3d(-30,30)
ax.set_zlim3d(0,60)
ax.view_init(10,5,'y')
plt.savefig('experiment/shape_registration/result'+exp_num+n_point+per_s+'/icp_umeyama/'+'init'+'.jpg')
plt.show()
plt.close()



n_iteration=400

# initlize 
rotation=torch.eye(3,dtype=torch.float32)
scalar=1.0 #

beta=(torch.mean(X1,0)-torch.mean(scalar*Y1@rotation,0))*(9.5/9)

paramlist=[]
X1_hat=Y1T@rotation*scalar+beta   


mass_diff=0
for epoch in range(n_iteration):
    M=cost_matrix_T(X1_hat,X1)
    argmin_X1=M.argmin(dim=1)
    X1_take=X1[argmin_X1]
    X1_hat_take=X1_take
    rotation,scalar=recover_rotation(X1_hat_take,Y1)
    #scalar=torch.mean(scalar_d)
    beta=torch.mean(X1_hat_take,0)-torch.mean(scalar*Y1@rotation,0)
    X1_hat=Y1T@rotation*scalar+beta
    
    param={}
    param['rotation']=rotation
    param['beta']=beta
    param['scalar']=scalar
    paramlist.append(param)

    
    if epoch<=200 or epoch%20==0 or epoch==n_iteration-1:
        print('training Epoch {}/{}'.format(epoch, n_iteration))
        print('scalar',scalar)
        print('rotation',rotation)
        print('beta',beta)
    
        X1_hat_c=X1_hat.clone().detach().cpu()
        fig = plt.figure(figsize=(10,10))
        ncolors = len(plt.rcParams['axes.prop_cycle'])
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
        ax.scatter(X1_hat_c[:,0],X1_hat_c[:,1],X1_hat_c[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
        plt.axis('off')
        ax.set_facecolor("grey")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        plt.legend(loc='upper right',scatterpoints=100)

        ax.set_xlim3d(-45,45)
        ax.set_ylim3d(-30,30)
        ax.set_zlim3d(0,60)
        ax.view_init(10,5,'y')

        plt.savefig('experiment/shape_registration/result/'+exp_num+n_point+per_s+'/icp_umeyama/'+str(epoch)+'.jpg')
        plt.show()
        plt.close()
        print('-' * 10)
    

torch.save(paramlist,'experiment/shape_registration/result/'+exp_num+n_point+per_s+'/icp_umeyama_param.pt')


    
    
    
    

 