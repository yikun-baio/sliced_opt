# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:58:08 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu 
"""
import os
import numpy as np
from typing import Tuple
import torch
from scipy.stats import ortho_group
import sys
import numba as nb
from .library import *
from .lib_ot import *



def random_projections_T(d,n_projections,Type): #,device='cpu',dtype=torch.float,Type=None):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
    if Type==0:
#        torch.manual_seed(0)
        Gaussian_vector=np.random.normal(0,1,size=[d,n_projections],device=device,dtype=dtype)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T
    elif Type==1:
#        np.random.seed(0)
        r=int(n_projections/d)+1
        projections=np.concatenate([ortho_group.rvs(d) for i in range(r)],axis=1)
        projections=projections[0:n_projections]
#        projections=torch.from_numpy(projections).to(device=device).to(dtype=dtype).T
    else:
        print('Type must be None or orth')
    return projections




@nb.njit(['float64[:,:](int64,int64,int64)'],fastmath=True,cache=True)
def random_projections(d,n_projections,Type=0):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
#    np.random.seed(0)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)) #.astype(np.float64)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((d*r,d)) #,dtype=np.float64)
        for i in range(r):
            H=np.random.randn(d,d) #.astype(np.float64)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections


@nb.njit(['float32[:,:](int64,int64,int64)'],fastmath=True,cache=True)
def random_projections_32(d,n_projections,Type=0):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
    np.random.seed(0)
    if Type==0:
        Gaussian_vector=np.random.normal(0,1,size=(d,n_projections)).astype(np.float32) #.astype(np.float64)
        projections=Gaussian_vector/np.sqrt(np.sum(np.square(Gaussian_vector),0))
        projections=projections.T

    elif Type==1:
        r=np.int64(n_projections/d)+1
        projections=np.zeros((r*d,d),dtype=np.float32)
        for i in range(r):
            H=np.random.randn(d,d).astype(np.float32)
            Q,R=np.linalg.qr(H)
            projections[i*d:(i+1)*d]=Q
        projections=projections[0:n_projections]
    return projections


#@nb.njit([nb.types.Tuple((nb.float64[:],nb.int64[:,:]))(nb.float64[:,:],nb.float64[:,:],nb.float64)],parallel=True,fastmath=True)
@nb.njit(['Tuple((float64[:],int64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True,cache=True)
def opt_plans(X_sliced,Y_sliced,Lambda_list):
    N,n=X_sliced.shape
#    Dtype=type(X_sliced[0,0])
    plans=np.zeros((N,n),np.int64)
    costs=np.zeros(N,np.float64)
    for i in nb.prange(N):
        X_theta=X_sliced[i]
        Y_theta=Y_sliced[i]
        Lambda=Lambda_list[i]
        M=cost_matrix(X_theta,Y_theta)
        obj,phi,psi,piRow,piCol=solve_opt(M,Lambda)
        cost=obj
        L=piRow
        plans[i]=L
        costs[i]=cost
    return costs,plans





@nb.njit(['(float64[:,:],float64[:,:],float64[:,:],float64[:])'],cache=True)
def X_correspondence(X,Y,projections,Lambda_list):
    N,d=projections.shape
    n=X.shape[0]
    Lx_org=arange(0,n)
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        Lambda=Lambda_list[i]
        M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt(M,Lambda)
#        Cost,L=o(X_s,Y_s,Lambda)
        
        L=piRow
        L=recover_indice(X_indice,Y_indice,L)
        #move X
        Lx=Lx_org.copy()
        Lx=Lx[L>=0]
        if Lx.shape[0]>=1:
            Ly=L[L>=0]
#            dim=Ly.shape[0]
            X_take=X_theta[Lx]
            Y_take=Y_theta[Ly]
            X[Lx]+=np.expand_dims(Y_take-X_take,1)*theta
            
@nb.njit(['(float32[:,:],float32[:,:],float32[:,:],float32[:])'],cache=True)
def X_correspondence_32(X,Y,projections,Lambda_list):
    N,d=projections.shape
    n=X.shape[0]
    Lx_org=arange(0,n)
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        Lambda=Lambda_list[i]
        M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt_32(M,Lambda)
        L=piRow
        L=recover_indice(X_indice,Y_indice,L)
        #move X
        Lx=Lx_org.copy()
        Lx=Lx[L>=0]
        if Lx.shape[0]>=1:
            Ly=L[L>=0]
#            dim=Ly.shape[0]
            X_take=X_theta[Lx]
            Y_take=Y_theta[Ly]
            X[Lx]+=np.expand_dims(Y_take-X_take,1)*theta

 

@nb.njit(['(float64[:,:],float64[:,:],float64[:,:])'],cache=True)
def X_correspondence_pot(X,Y,projections):
    N,d=projections.shape
    n=X.shape[0]
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        M=cost_matrix(X_s,Y_s)
        cost,L=pot(M)
        L=recover_indice(X_indice,Y_indice,L)
        X_take=X_theta
        Y_take=Y_theta[L]
        X+=np.expand_dims(Y_take-X_take,1)*theta
    return X

@nb.njit(['(float32[:,:],float32[:,:],float32[:,:])'])
def X_correspondence_pot_32(X,Y,projections):
    N,d=projections.shape
    n=X.shape[0]
    for i in range(N):
        theta=projections[i]
        X_theta=np.dot(theta,X.T)
        Y_theta=np.dot(theta,Y.T)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        M=cost_matrix(X_s,Y_s)
        cost,L=pot_32(M)
        L=recover_indice(X_indice,Y_indice,L)
        X_take=X_theta
        Y_take=Y_theta[L]
        X+=np.expand_dims(Y_take-X_take,1)*theta
    return X

    

@nb.njit(['Tuple((float64,int64[:,:],float64[:,:],float64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True,cache=True)
def opt_plans_64(X,Y,Lambda_list):
    n,d=X.shape
    n_projections=Lambda_list.shape[0]
    projections=random_projections(d,n_projections,0)
    X_projections=projections.dot(X.T)
    Y_projections=projections.dot(Y.T)
    opt_plan_X_list=np.zeros((n_projections,n),dtype=np.int64)
    #opt_plan_Y_list=np.zeros((n_projections,n),dtype=np.int64)
    opt_cost_list=np.zeros(n_projections)
    for (epoch,(X_theta,Y_theta,Lambda)) in enumerate(zip(X_projections,Y_projections,Lambda_list)):
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        M=cost_matrix(X_s,Y_s)
        obj,phi,psi,piRow,piCol=solve_opt(M,Lambda)
        
        L1=recover_indice(X_indice,Y_indice,piRow)
        #L2=recover_indice(Y_indice,X_indice,piCol)
        opt_cost_list[epoch]=obj
        opt_plan_X_list[epoch]=L1
        #opt_plan_Y_list[epoch]=L2
        #sopt_dist=np.sum(opt_cost_list)/n_projections
        sopt_dist=opt_cost_list.sum()/n_projections
    return sopt_dist,opt_plan_X_list,X_projections,Y_projections

def opt_cost_from_plans(X_projections,Y_projections,Lambda_list,opt_plan_X_list,cache=True):
    n_projections,n=X_projections.shape
    n_projections,m=Y_projections.shape
    opt_cost_list=np.zeros(n_projections)
    for (epoch,(X_theta,Y_theta,Lambda,opt_plan)) in enumerate(zip(X_projections,Y_projections,Lambda_list,opt_plan_list)):
        Domain=opt_plan>=0
        Range=opt_plan[Domain]
        X_select=X_theta[Domain]
        Y_select=Y_theta[Range]
        trans_cost=np.sum(cost_function(X_select,Y_select))
        mass_panalty=Lambda*(m+n-2*Domain.sum())
        opt_cost_list[epoch]=trans_cost+mass_panalty
    return opt_cost_list


        
        
   
        

    

    
    
    
    


    
    
    
    
        
    

    
    




    


