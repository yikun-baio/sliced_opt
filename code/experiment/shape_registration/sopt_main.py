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
#'/mumble_sitting'
#'/dragon' 
#'/stanford_bunny' #'/witchcastle' #'mumble_sitting'
exp_num=item
#'/mumble_sitting'
#'/dragon' 
#'/stanford_bunny'#'/witchcastle' #'mumble_sitting'

from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   
label_L=['0','1','2','3']
L=['/10k','/9k','/8k','/7k']
for (label,per_s) in [('0','-5p')]:
    n_point=L[int(label)]
    
    data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
    save_path='experiment/shape_registration/result'+exp_num+n_point
    data=torch.load(data_path+item+'.pt')
    
    
    X0=data['X0'].to(torch.float32)
    Y0=data['Y0'+label].to(torch.float32)
    X1=data['X1'+per_s].to(torch.float32)
    Y1=data['Y1'+label+per_s].to(torch.float32)
    print('Y1.shape is', Y1.shape)
    #param=data['param']
    
    
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
    
    # ax.view_init(15,15,'y')
    plt.savefig('experiment/shape_registration/result'+exp_num+n_point+per_s+'/sopt/'+'init'+'.jpg')
    plt.show()
    plt.close()
    
    
    n_iteration=4000
    
    # initlize 
    rotation=torch.eye(3,dtype=torch.float32)
    scalar=1.0 #
    
    beta=(torch.mean(X1,0)-torch.mean(scalar*Y1@rotation,0))
    
    paramlist=[]
    #n_projections=1
    Lambda=np.float32(3*torch.sum(beta**2).item())
    Delta=Lambda*1/8
    X1_hat=Y1T@rotation*scalar+beta   
    
    A=sopt(X1_hat,X1T,Lambda,n_iteration,'orth')
    #A=sopt_correspondence(X1_hat,X1T,Lambda,50,'orth')
    mass_diff=0
    N0=Y0.shape[0]
    N1=Y1.shape[0]
    b=np.log((N1-N0)/1)
    N=N0
    
    for epoch in range(n_iteration):
        A.get_directions()
        A.get_one_projection(epoch)
        A.get_plans()
        loss,mass=A.sliced_cost()
       
        n=A.X_take.shape[0]
        A.X[A.Lx]+=(A.Y_take-A.X_take).reshape((n,1))*A.projections[epoch]
        #extract the paired data
        Y1_take=Y1T[A.Lx]
        X1_hat_take=A.X[A.Lx]
        
        # A.get_directions()
        # A.correspond(N)
        # Y1_take=Y1T[A.Lx]
        # X1_hat_take=A.X[A.Lx]
        
        rotation,scalar_d=recover_rotation(X1_hat_take,Y1_take)
        #scalar=torch.mean(scalar_d)
        scalar=torch.sqrt(torch.trace(torch.cov(X1_hat_take.T))/torch.trace(torch.cov(Y1_take.T)))
        beta=torch.mean(X1_hat_take,0)-torch.mean(scalar*Y1_take@rotation,0)
        A.X=Y1T@rotation*scalar+beta
        

        #N=(N1-N0)*np.exp(-epoch/n_iteration*b)+N0
        #N=(N1-N0)*1/(1+b*(epoch/n_iteration))+N0
    
        mass_diff=mass.item()-N
    
        
        if mass_diff>N*0.009:
            A.Lambda-=Delta
          
        if mass_diff<-N*0.003:
            A.Lambda+=Delta
            Delta=A.Lambda*1/8
        if A.Lambda<=Delta:
            A.Lambda=Delta
            Delta=A.Lambda/10 
            
    
        
        
        param={}
        param['rotation']=rotation
        param['beta']=beta
        param['scalar']=scalar
        paramlist.append(param)
    
        
        if epoch<=200 or epoch%20==0 or epoch==n_iteration-1:
            print('N is',N)
            print('training Epoch {}/{}'.format(epoch, n_iteration))
            print('lambda',A.Lambda)
            print('mass_diff',mass_diff)
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
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlim3d(-0.08,0.12)
            ax.set_ylim3d(0.06,0.2)
            ax.set_zlim3d(-0.02,0.14)
    
            ax.view_init(10,5,'y')
            
            plt.legend(loc='upper right',scatterpoints=100)
            plt.savefig('experiment/shape_registration/result/'+exp_num+n_point+per_s+'/sopt/'+str(epoch)+'.jpg')
            plt.show()
            plt.close()
            print('-' * 10)
        
        # pcd_X1_hat= o3d.geometry.PointCloud()
        # pcd_X1_hat.points = o3d.utility.Vector3dVector(X1_hat_c.numpy())
        # pcd_X1.paint_uniform_color([0.1, 0.1, 1])
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=False) #works for me with False, on some systems needs to be true
        # vis.add_geometry(pcd_X1)
        # vis.add_geometry(pcd_X1_hat)
        # vis.reset_view_point(100)
        # vis.get_render_option().background_color=np.array([0.6, 0.6, 0.6])
        # ctr = vis.get_view_control()
        # ctr.rotate(0,-400,0)
        # vis.poll_events()
        # vis.update_renderer()
        # #vis.capture_screen_image(save_root+'/test1.png')
        # vis.capture_screen_image(save_path+'/sopt/'+str(epoch)+'.jpg' )
        # vis.destroy_window()
        
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X1_hat_c[:,0],X1_hat_c[:,1],X1_hat_c[:,2],s=0.3,label='source') # plot the point (2,3,4) on the figure
        # ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=0.3,label='target') # plot the point (2,3,4) on the figure
        # ax.set_xlim3d(-30,30)
        # ax.set_ylim3d(-15,15)
        # ax.set_zlim3d(0,35)
        
        # ax.set_xlim3d(-30,40)
        # ax.set_ylim3d(-40,40)
        # ax.set_zlim3d(0,60)
        # plt.legend(loc='upper right')
        # plt.savefig('experiment/shape_registration/result/'+exp_num+n_point+'/sopt/'+str(epoch)+'.jpg')
        # plt.show()
        # plt.close()
        
      
        #print('loss is ',loss.item())
    
    
    torch.save(paramlist,'experiment/shape_registration/result/'+exp_num+n_point+per_s+'/sopt_param.pt')
