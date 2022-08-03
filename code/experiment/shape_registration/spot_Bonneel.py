#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
item='/stanford_bunny'
#'/stanford_bunny' #'/witchcastle' #'mumble_sitting'
exp_num='/stanford_bunny' 
#'/stanford_bunny'#'/witchcastle' #'mumble_sitting'

from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   
label_L=['0','1','2','3']
L=['/10k','/9k','/8k','/7k']
for (label,per_s) in [('0','-7p'),('1','-5p'),('1','-7p')]:
    n_point=L[int(label)] #'/9k'
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
    ax.set_xlim3d(-0.08,0.12)
    ax.set_ylim3d(0.06,0.2)
    ax.set_zlim3d(-0.02,0.14)
    ax.view_init(10,5,'y')
    plt.legend(loc='upper right',scatterpoints=100)
    #ax.view_init(10,5,'y')
    plt.savefig('experiment/shape_registration/result'+exp_num+n_point+per_s+'/spot/'+'init'+'.jpg')
    plt.show()
    plt.close()
    
    
    start_time=time.time()
    n_iteration=200
    
    # initlize 
    rotation=torch.eye(3,dtype=torch.float32)
    scalar=1.0 #
    beta=(torch.mean(X1,0)-torch.mean(scalar*Y1@rotation,0))
    paramlist=[]
    #n_projections=1
    Lambda=np.float32(200)
    Delta=Lambda*0.1
    X1_hat=Y1T@rotation*scalar+beta   
    
    #A=sopt(X1_hat,X1T,Lambda,n_iteration,'orth')
    A=sopt_correspondence(X1_hat,X1T,Lambda,100,'orth')
    mass_diff=0
    for epoch in range(n_iteration):
        
        A.get_directions()
        A.correspond(N)
        Y1_take=Y1T
        X1_hat_take=A.X
        rotation,scalar=recover_rotation(X1_hat_take,Y1_take)
     #   scalar=torch.sqrt(torch.trace(torch.cov(X1_hat_take.T))/torch.trace(torch.cov(Y1_take.T)))
        beta=torch.mean(X1_hat_take,0)-torch.mean(scalar*Y1_take@rotation,0)
        A.X=Y1T@rotation*scalar+beta
        
        
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
        
            X1_hat_c=A.X.clone().detach().cpu()
            
            fig = plt.figure(figsize=(10,10))
            ncolors = len(plt.rcParams['axes.prop_cycle'])
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
            ax.scatter(X1_hat_c[:,0],X1_hat_c[:,1],X1_hat_c[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
            plt.axis('off')
            ax.set_facecolor("grey")
            ax.grid(False)
            ax.set_xlim3d(-0.08,0.12)
            ax.set_ylim3d(0.06,0.2)
            ax.set_zlim3d(-0.02,0.14)
            ax.view_init(10,5,'y')
    #        ax.view_init(10,5,'y')
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_zticks([])
            # ax.view_init(10,5,'y')
            
            plt.legend(loc='upper right',scatterpoints=100)
    
            plt.savefig('experiment/shape_registration/result/'+exp_num+n_point+per_s+'/spot/'+str(epoch)+'.jpg')
            plt.show()
            plt.close()
            print('-' * 10)
        
    
    
    torch.save(paramlist,'experiment/shape_registration/result/'+exp_num+n_point+per_s+'/spot_param.pt')
