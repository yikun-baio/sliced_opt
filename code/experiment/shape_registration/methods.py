#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:01:32 2022

@author: baly
"""



import sys
import open3d as o3d
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import numpy as np
import ot
import time
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   

# our method
@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64,nb.int64)])
def sopt_main(X,Y,n_iterations,N0):
    n,d=X.shape
    N1=Y.shape[0]
    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1.0) #
    beta=vec_mean(X)-vec_mean(scalar*Y.dot(rotation))
    #paramlist=[]
    projections=random_projections_nb(d,n_iterations,1)
    mass_diff=0
    b=np.log((N1-N0)/1)
    Lambda=3*np.sum(beta**2)
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    X_hat=Y.dot(rotation)*scalar+beta
    
    Lx_hat_org=arange(0,n)
    Delta=Lambda/8
    lower_bound=Lambda/100
    for i in range(n_iterations):
        theta=projections[i]
        X_hat_theta=theta.dot(X_hat.T)
        X_theta=theta.dot(X.T)

        X_hat_indice=X_hat_theta.argsort()
        X_indice=X_theta.argsort()
        X_hat_s=X_hat_theta[X_hat_indice]
        X_s=X_theta[X_indice]
        cost,L=opt_1d_v2_apro(X_hat_s,X_s,Lambda)
        L=recover_indice(X_hat_indice,X_indice,L)
        
        #move Xhat
        Lx_hat=Lx_hat_org.copy()
        Lx_hat=Lx_hat[L>=0]
        mass=Lx_hat.shape[0]
        if Lx_hat.shape[0]>=1:
            Lx=L[L>=0]
            X_hat_take=X_hat_theta[Lx_hat]
            X_take=X_theta[Lx]
            X_hat[Lx_hat]+=transpose(X_take-X_hat_take)*theta
        
        X_hat_take=X_hat[Lx_hat]
        Y_take=Y[Lx_hat]
        
        rotation,scalar_d=recover_rotation_nb(X_hat_take,Y_take)
        scalar=np.float32(np.sqrt(np.trace(np.cov(X_hat_take.T))/np.trace(np.cov(Y_take.T))))
        beta=vec_mean(X_hat_take)-vec_mean(scalar*Y_take.dot(rotation))
        X_hat=Y.dot(rotation)*scalar+beta
        
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta
        
        N=(N1-N0)*1/(1+b*(i/n_iterations))+N0
    
        mass_diff=mass-N
    
        
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<lower_bound:
            Lambda=lower_bound
        
        # fig = plt.figure(figsize=(10,10))
        # ncolors = len(plt.rcParams['axes.prop_cycle'])
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X[:,0],X[:,1],X[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
        # ax.scatter(X_hat[:,0],X_hat[:,1],X_hat[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
        # plt.axis('off')
        # ax.set_facecolor("grey")
        # ax.grid(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_xlim3d(-45,45)
        # ax.set_ylim3d(-30,30)
        # ax.set_zlim3d(0,60)
        # ax.view_init(10,5,'y')
        # plt.legend(loc='upper right',scatterpoints=100)
        
        # ax.view_init(15,15,'y')
#        plt.savefig('experiment/shape_registration/result'+exp_num+n_point+per_s+'/sopt/'+'init'+'.jpg')
        # plt.show()
        # plt.close()

    return rotation_list,scalar_list,beta_list    

# method of spot_boneel method
@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64,nb.int64)])
def spot_bonneel(X,Y,n_projections,n_iterations):
    n,d=X.shape
    N1=Y.shape[0]
    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1.0) #
    beta=vec_mean(X)-vec_mean(scalar*Y.dot(rotation))
    #paramlist=[]
    
    
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    X_hat=Y.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
        projections=random_projections_nb(d,n_projections,1)
        X_correspondence_pot(X_hat,X,projections)
        
        rotation,scalar=recover_rotation_nb(X_hat,Y)
        beta=vec_mean(X_hat)-vec_mean(scalar*Y.dot(rotation))
        X_hat=Y.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list    


@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64)])
def icp_du(X,Y,n_iterations):
    n,d=X.shape

    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1.0) #
    beta=vec_mean(X)-vec_mean(scalar*Y.dot(rotation))
    #paramlist=[]
    
    
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    X_hat=Y.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
        M=cost_matrix_d(X_hat,X)
        argmin_X=closest_y_M(M)
        X_take=X[argmin_X]
        X_hat=X_take
        rotation,scalar_d=recover_rotation_du_nb(X_hat,Y)
        scalar=np.mean(scalar_d)
        beta=vec_mean(X_hat)-vec_mean(scalar*Y.dot(rotation))
        X_hat=Y.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  



@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64)])
def icp_umeyama(X,Y,n_iterations):
    n,d=X.shape

    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1.0) #
    beta=vec_mean(X)-vec_mean(scalar*Y.dot(rotation))
    #paramlist=[]
    
    
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    X_hat=Y.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
        M=cost_matrix_d(X_hat,X)
        argmin_X=closest_y_M(M)
        X_take=X[argmin_X]
        X_hat=X_take
        rotation,scalar=recover_rotation_nb(X_hat,Y)
        #scalar=np.mean(scalar_d)
        beta=vec_mean(X_hat)-vec_mean(scalar*Y.dot(rotation))
        X_hat=Y.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  

