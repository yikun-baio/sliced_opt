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




@nb.njit(['float64[:,:](int64,int64,int64)'],fastmath=True)
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


@nb.njit(['float32[:,:](int64,int64,int64)'],fastmath=True)
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
@nb.njit(['Tuple((float64[:],int64[:,:]))(float64[:,:],float64[:,:],float64[:])'],parallel=True,fastmath=True)
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





@nb.njit(['(float64[:,:],float64[:,:],float64[:,:],float64[:])'])
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
            
@nb.njit(['(float32[:,:],float32[:,:],float32[:,:],float32[:])'])
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

 

@nb.njit(['(float64[:,:],float64[:,:],float64[:,:])'])
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

    



        
        
   
        
        
        

        
    

    
    
    
    


    
    
    
    
        
    

    
    




    


