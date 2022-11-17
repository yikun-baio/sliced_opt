#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr 19 11:32:17 2022

# @author: laoba
# """

import numpy as np
import torch
import os

import numba as nb
from typing import Tuple #,List
from numba.typed import List


@nb.njit()
def cost_function(x,y): 
    ''' 
    case 1:
        input:
            x: float number
            y: float number 
        output:
            (x-y)**2: float number 
    case 2: 
        input: 
            x: n*1 float np array
            y: n*1 float np array
        output:
            (x-y)**2 n*1 float np array, whose i-th entry is (x_i-y_i)**2
    '''
#    V=np.square(x-y) #**p
    V=np.power(x-y,2)
    return V


@torch.jit.script
def cost_function_T(x,y): 
    ''' 
    case 1:
        input:
            x: 0 dimension float tensor
            y: 0 dimension float tensor
        output:
            float number: (x-y)**2 
    case 2: 
        input: 
            x: n*1 tensor
            y: n*1 tensor 
        output:
            n*1 array: whose ith entry is (x_i-y_i)**2
    '''
    return torch.square(x-y)

@nb.njit(nb.float32[:,:](nb.float32[:]),fastmath=True)
def transpose(X):
    n=X.shape[0]
    XT=np.zeros((n,1),dtype=np.float32)
    for i in range(n):
        XT[i]=X[i]
    return XT

#@nb.jit(nopython=True)
@nb.njit(nb.float32[:,:](nb.float32[:],nb.float32[:]))
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    XT=transpose(X)


    M=cost_function(XT,Y)
    return M


    


#@nb.jit(nopython=True)
@nb.njit(nb.float32[:,:](nb.float32[:,:],nb.float32[:,:]),fastmath=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    n=X.shape[0]
    m=Y.shape[0]
    M=np.zeros((n,m),dtype=nb.float32)
    for i in range(n):
        for j in range(m):
            M[i,j]=np.sum(cost_function(X[i,:],Y[j,:]))
    return M


@nb.njit(nb.float32[:](nb.float32[:,:],nb.float32[:]),fastmath=True)
def mat_vec_mul(XT,theta):
    d,n=XT.shape 
    result=np.zeros(n,dtype=np.float32)
    for i in range(n):
        result[i]=np.dot(XT[:,i],theta)
    return result
    

#@nb.jit([float32[:,:](float32[:],float32[:])],forceobj=True)
@torch.jit.script
def cost_matrix_T(X,Y):
    '''
    input: 
        X: n*d float torch tensor
        Y: m*d float torch tensor
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function_T.
    
    '''
    if len(X.shape)==1:
        X=X.reshape([X.shape[0],1])
        M=cost_function_T(X,Y)
    else:
        device=X.device.type
        n,d=X.shape
        m=Y.shape[0]
        M=torch.zeros([n,m],device=device)
        for i in range(d):
            M+=cost_function_T(X[:,i:i+1],Y[:,i:i+1].T)      
#        M=torch.sum(torch.stack([cost_function_T(X[:,d:d+1],Y[:,d:d+1].T) for d in range(dim)]),0)
    return M



#types.Tuple

#@nb.jit([float32(float32,float32)],forceobj=True) 
@nb.jit(nopython=True)
def closest_y(x,Y):
    '''
    Parameters
    ----------
    x : float number, xk
    Y : m*1 float np array, 
    Returns
    -------
    min_index : integer >=0
        argmin_j min(x,Y[j])  # you can also return 
    min_cost : float number 
        Y[min_index]
    '''
    cost_list=cost_function(x,Y)    
    min_index=cost_list.argmin()
    min_cost=cost_list[min_index]
    return min_index,min_cost

@nb.njit(fastmath=True)
def closest_y_M(M):
    '''
    Parameters
    ----------
    x : float number, xk
    Y : m*1 float np array, 
    Returns
    -------
    min_index : integer >=0
        argmin_j min(x,Y[j])  # you can also return 
    min_cost : float number 
        Y[min_index]
    '''
    n,m=M.shape
    argmin_Y=np.zeros(n,dtype=np.int64)
    for i in range(n):
        argmin_Y[i]=M[i,:].argmin()
        
    return argmin_Y



@nb.njit([(nb.int64[:],nb.int64)])  
def index_adjust(L,j_start=0):
    '''
    Parameters
    ----------
    L : np n*1 array, whose entry is 0,1,2,..... or -1. 
          transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
          if we ignore -1, L must be in increasing order 
        
    j_start : integer>=0 
          index of y 
    Returns
    -------
    L : List, whose entry is 0,1,.... or -1 
    Go through all the entries in L, if the entry>=0, add j_start,
    eg. input (L=[1,2,3], j_start=2)
        return L=[3,4,5]
    eg. input (L=[1,2,-1,3], j_start=2)
        return L=[3,4,-1,5]
    '''
    positive_indices=(L>=0).nonzero()
    L[positive_indices]=L[positive_indices]+j_start

         

@torch.jit.script   
def index_adjust_T(L: torch.Tensor,j_start: int =0):
    '''
    Parameters
    ----------
    L : n*1 torch tensor, whose entry is 0,1,2,3..... or -1. 
          transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
          if we ignore -1, L must be in increasing order 
    j_start : integer>=0 
          index of y 
    Returns
    -------
    L : List, whose entry is 0,1,.... or -1 
    Go through all the entries in L, if the entry>=0, add j_start,
    eg. input (L=[1,2,3], j_start=2)
        return L=[3,4,5]
    eg. input (L=[1,2,-1,3], j_start=2)
        return L=[3,4,-1,5]
    '''
    
    positive_indices=(L>=0).nonzero()
    L[positive_indices]=L[positive_indices]+j_start
    return None

@nb.jit(nb.types.Tuple((nb.int64,nb.int64))(nb.int64[:]),nopython=True)  
def startindex(L_pre):
    '''
    Parameters
    ----------
    L_pre: n*1 np array, whose entry is 0,1,2,..... or -1. 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L_pre must be in increasing order 
            make sure L_pre[end]=-1, or L_pre=[] otherwise there is mistake in the main loop.  
         
         
    Returns
    -------
    i_start : integer>=0 
        start point of X
    j_start : integer>=0
        start point of Y
      
    if L_pre=[], then i_start=0, j_start=0
    otherwise:
    i_start=len(L_pre)
    j_start=max{j: j in L, j>=0}+1
    
    Ex1: input: L_pre=[]
    return: 0,0 
    input: L_pre=[0,2,5-1]
    return: 4,6
    input: L_pre=[0,-1,2,4,-1,-1]
    return: 6,5
    '''
    i_start=L_pre.shape[0]
    length=L_pre.shape[0]
    if i_start==0:
        return 0,0
    for i in range(length-1,-1,-1):
        j=L_pre[i]
        if j>=0:
            j_start=j+1
            return i_start,j_start

    return i_start,0


@nb.jit(nb.types.Tuple((nb.int64,nb.int64))(nb.int64[:]),nopython=True)
def startindex_np(L_pre):
    '''
    Parameters
    ----------
    L_pre: n*1 np array, whose entry is 0,1,2,..... or -1. 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L_pre must be in increasing order 
            make sure L_pre[end]=-1, or L_pre=[] otherwise there is mistake in the main loop.  
         
         
    Returns
    -------
    i_start : integer>=0 
        start point of X
    j_start : integer>=0
        start point of Y
      
    if L_pre=[], then i_start=0, j_start=0
    otherwise:
    i_start=len(L_pre)
    j_start=max{j: j in L, j>=0}+1
    
    Ex1: input: L_pre=[]
    return: 0,0 
    input: L_pre=[0,2,5-1]
    return: 4,6
    input: L_pre=[0,-1,2,4,-1,-1]
    return: 6,5
    '''
    
    i_start=L_pre.shape[0]
    if i_start==0:
        return 0,0
    j_start=max(0,L_pre.max()+1)
    return i_start,j_start

@nb.jit([nb.int64[:](nb.int64,nb.int64)],nopython=True)
def arange(start,end):
    n=end-start
    L=np.zeros(n,dtype=np.int64)
    for i in range(n):
        L[i]=i+start
    return L



@torch.jit.script   
def startindex_T(L_pre):    
      '''
      Parameters
      ----------
      L_pre: n*1 torch tensor, whose entry is 0,1,2,..... or -1. 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L_pre must be in increasing order 
            make sure L_pre[end]=-1, or L_pre=[] otherwise there is mistake in the main loop.  
          
      Returns
      -------
      i_start : integer>=0 
        start point of X
      j_start : integer>=0
        start point of Y
        
      if L_pre=[], i_start=0, j_start=0
      otherwise:
      i_start=len(L_pre)
      j_start=max{j: j in L, j>=0}+1
     
      Ex1: input: L_pre=[]
      return: 0,0 
      input: L_pre=[0,2,5-1]
      return: 4,6
      input: L_pre=[0,-1,2,4,-1,-1]
      return: 6,5
    
    
      '''
      i_start=L_pre.shape[0]
      device=L_pre.device.type
      if i_start==0:
          return 0,0
      #L_pre_assign=L_pre[L_pre>=0]
      #if L_pre_assign.shape[0]==0:
      #    return i_start,0
      #else:        
#    j_start=L_pre_assign[-1].item()+1
      j_start=max(0,L_pre.max()+1)
      return i_start,j_start





@nb.jit([nb.types.Tuple((nb.int64,nb.int64))(nb.int64[:])],nopython=True)
def unassign_y(L1):
    '''
    Parameters
    ----------
    L1 : n*1 list , whose entry is 0,1,2,...... 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L1 must be in increasing order 
            make sure L1 do not have -1 and is not empty, otherwise there is mistake in the main loop.  
    Returns
    -------
    i_act: integer>=0 
    j_act: integer>=0 or -1    
    j_act=max{j: j not in L1, j<L1[end]} If L1[end]=-1, there is a bug in the main loop. 
    i_act=min{i: L[i]>j_act}.
    
    Eg. input: L1=[1,3,5]
    return: 2,4
    input: L1=[2,3,4]
    return: 0,1
    input: L1=[0,1,2,3]
    return: 0,-1
    
    '''
    n=L1.shape[0]
    j_last=L1[n-1]
    i_last=L1.shape[0]-1 # this is the value of k-i_start
    for l in range(n):
        j=j_last-l
        i=i_last-l+1
        if j > L1[n-1-l]:
            return i,j
    j=j_last-n
    if j>=0:
        return 0,j
    else:       
        return 0,-1



@torch.jit.script   
def unassign_y_T(L1) -> Tuple[int,torch.Tensor]:
    '''
    Parameters
    ----------
    L1 : n*1 torch tensor, whose entry is 0,1,2,...... 
            transporportation plan. L[i]=j denote we assign x_i to y_j, L[i]=-1, denote we destroy x_i. 
            if we ignore -1, L1 must be in increasing order 
            make sure L1 do not have -1 and is not empty, otherwise there is mistake in the main loop.  
           
    Returns
    -------
    i_act: integer>=0 
    j_act: integer>=0 or -1    
    j_act=max{j: j not in L1, j<L1[end]} If L1[end]=-1, there is a bug in the main loop. 
    i_act=min{i: L[i]>j_act}.
    
    eg. input: L1=[1,3,5]
    return: 2,4
    eg. input: L1=[2,3,4]
    return: 0,1
    eg. input: L1=[0,1,2,3]
    return: 0,-1
    
    '''
    device=L1.device.type
    j_last=L1[-1]
    n=L1.shape[0]
    L_range=torch.arange(j_last-n+1,j_last+1,device=device)
    L_dif=torch.where(L_range-L1>0)[0]
    if L_dif.shape[0]==0:
        return 0, L1[0]-1   
    i_act=L_dif[-1]+1
    j_act=L_range[i_act-1]
    return i_act.item(),j_act

# not test

@torch.jit.script   
def recover_indice_T(indice_X,indice_Y,L):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    device=indice_X.device.type
    n=L.shape[0]
#    indice_Y_mapped=torch.tensor([indice_Y[i] if i>=0 else -1 for i in L],device=device)
    indice_Y_mapped=torch.where(L>=0,indice_Y[L],-1).to(device) 
    mapping=torch.stack([indice_X,indice_Y_mapped])
    mapping_final=mapping[1].take(mapping[0].argsort())
    return mapping_final

#@torch.jit.script   
@nb.njit(nb.int64[:](nb.int64[:],nb.int64[:],nb.int64[:]))
def recover_indice(indice_X,indice_Y,L):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    n=L.shape[0]
    indice_Y_mapped=np.where(L>=0,indice_Y[L],-1)
    mapping=np.stack((indice_X,indice_Y_mapped))
    mapping_final=mapping[1].take(mapping[0].argsort())
    return mapping_final


@torch.jit.script   
def recover_indice_M(indice_X,indice_Y,plans):
    '''
    input:
        indice_X: n*1 float torch tensor, whose entry is integer 0,1,2,....
        indice_Y: m*1 float torch tensor, whose entry is integer 0,1,2,.... 
        L: n*1 list, whose entry could be 0,1,2,... and -1.
        L is the original transportation plan for sorted X,Y 
        L[i]=j denote x_i->y_j and L[i]=-1 denote we destroy x_i. 
        If we ignore -1, it must be in increasing order  
    output:
        mapping_final: the transportation plan for original unsorted X,Y
        
        Eg. X=[2,1,3], indice_X=[1,0,2]
            Y=[3,1,2], indice_Y=[1,2,0]
            L=[0,1,2] which means the mapping 1->1, 2->2, 3->3
        return: 
            L=[2,1,0], which also means the mapping 2->2, 1->1,3->3.
    
    '''
    device=indice_X.device.type
    N,n=plans.shape
    indice_Y_mapped=torch.zeros((N,n),dtype=torch.int64,device=device)
    for i in range (N):
        indice_Y_mapped[i,:]=torch.where(plans[i]>=0,indice_Y[i].take(plans[i]),-1) 
#    indice_Y_mapped=torch.where(plans>=0,indice_Y.gather(1,plans),-1).to(device) 
    mapping=torch.stack([indice_X,indice_Y_mapped])
    mapping_final=torch.gather(mapping[1],1,mapping[0,:].argsort())
    return mapping_final


@torch.jit.script                        
def refined_cost_T(X,Y,L,Lambda,penulty: bool =False):
    '''
    Parameters
    ----------
    X : n*1 float tensor 
    Y : m*1 float tensor
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    Lambda : float number, determine the penualty 
    penulty : Booleans
        true denotes we add penulty to the return
        false denots we do not add penulty to the return
    Returns
    -------
    cost : 0 dimension float tensor 
        if penulty=False: sum_{i:L[i]>=0} c(X[i],Y[L[i]]) 
        if penulty=True: sum_{i:L[i]>=0}c(X[i],Y[L[i]])+Lambda*#{i:L[i]=-1} 
    '''
    n=L.shape[0]
    L_X=torch.arange(n)[L>=0]
    L_Y=L[L>=0]
    Y_take=Y[L_Y]
    X_take=X[L_X]
    
    if not penulty:
        cost=torch.sum(cost_function_T(X_take, Y_take))
    else:
        num_destruction=len(L)-len(L_X)
        cost=torch.sum(cost_function_T(X_take, Y_take))+Lambda*num_destruction
    return cost

@nb.njit()
def refined_cost(X,Y,L,Lambda,penulty=True):
    '''
    Parameters
    ----------
    X : n*1 float np array 
    Y : m*1 float np array
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    Lambda : float number, determine the penualty 
    penulty : Booleans
        true denotes we add penulty to the return
        false denots we do not add penulty to the return
    Returns
    -------
    cost : float number 
        if penulty=False: sum_{i:L[i]>=0} c(X[i],Y[L[i]]) 
        if penulty=True: sum_{i:L[i]>=0}c(X[i],Y[L[i]])+Lambda*#{i:L[i]=-1} 
    '''
    n=L.shape[0]
    Lx=arange(0,n)
    L_X=Lx[L>=0] #[i for i,j in enumerate(L) if j>=0]    
    L_Y=L[L>=0]
    Y_take=Y[L_Y]
    X_take=X[L_X]
    num_destruction=L.shape[0]-L_X.shape[0]
    if penulty==True:
        cost=np.sum(cost_function(X_take, Y_take))+Lambda*num_destruction
    else:
        cost=np.sum(cost_function(X_take, Y_take))
    return cost

@nb.njit([nb.float32[:,:](nb.int64[:],nb.int64)],fastmath=True)
def plan_to_matrix(L,m):
    '''
    Parameters
    ----------
    L : n*1 tensor, whose entries is 0,1,2,.... or -1
    
    m : integer >=0 
    
    Returns
    -------
    plan : n*m matrix
    plan[i,j]=1 if L[i]=j and j>=0
    otherwise, plan[i,j]=0
 
    '''
    n=L.shape[0]
    plan=np.zeros((n,m),dtype=np.float32)
    Ly=L[L>=0]
    Lx=arange(0,n)
    Lx=Lx[L>=0]
    for i in Lx:
            plan[i,L[i]]=1.0
    return plan
    


@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:],nb.float32,nb.int64[:]))(nb.int64,nb.float32)],nopython=True)
def empty_Y_opt(n,Lambda):
    '''
    Parameters
    ----------
    n : integer>=0
        size of X
    Lambda : float number >=0
    Returns
    -------
    cost : float number
        cost of the opt problem where Y is empty
    L : n*1 list. 
        transportation plan, whose entry could be 0,1,... or -1. 
        must be in increasing order
        in this function, it can only contains -1 
    cost_pre : float number 
        cost of the opt problem where Y is empty
        in this example, cost_pre=cost
    L_pre : n*1 list. 
        transportation plan, whose entry could be 0,1,... or -1. 
        must be in increasing order
        in this function, L_pre=L
    '''
    L=np.full(n,-1,dtype=np.int64) #(n,dtype=np.int64)+np.int64(-1)
    #for i in range(n):
    #    L[i]=-1
    cost=Lambda*np.float32(n)
    cost_pre=cost
    L_pre=L.copy()
    return cost,L,cost_pre,L_pre


# @nb.jit([nb.types.Tuple((nb.float32,nb.int64[:],nb.float32,nb.int64[:]))(nb.int64,nb.float32)],nopython=True)
# def empty_Y_opt_np(n,Lambda):
#     '''


#     Parameters
#     ----------
#     n : integer>=0
#         size of X
#     Lambda : float number >=0


#     Returns
#     -------
#     cost : float number
#         cost of the opt problem where Y is empty
#     L : n*1 np array. 
#         transportation plan, whose entry could be 0,1,... or -1. 
#         must be in increasing order
#         in this function, it can only contains -1 
#     cost_pre : float number 
#         cost of the opt problem where Y is empty
#         in this example, cost_pre=cost
#     L_pre : n*1 np array
#         transportation plan, whose entry could be 0,1,... or -1. 
#         must be in increasing order
#         in this function, L_pre=L

#     '''
#     L=np.zeros(n,dtype=np.int64)
#     for i in range(L.shape[0]):
#         L[i]=-1
#     cost=Lambda*n
#     cost_pre=cost
#     L_pre=L.copy()
#     return cost,L,cost_pre,L_pre


@torch.jit.script      
def empty_Y_opt_T(n: 'int',Lambda: 'torch.Tensor'):
    '''
    Parameters
    ----------
    n : integer>=0
        size of X
    Lambda : float number >=0
    Returns
    -------
    cost : float number
        cost of the opt problem where Y is empty
    L : n*1 torch tensor. 
        transportation plan, whose entry could be 0,1,... or -1. 
        must be in increasing order
        in this function, it can only contains -1 
    cost_pre : float number 
        cost of the opt problem where Y is empty
        in this example, cost_pre=cost
    L_pre : n*1 torch tensor. 
        transportation plan, whose entry could be 0,1,... or -1. 
        must be in increasing order
        in this function, L_pre=L
    '''
    device=Lambda.device.type
    L=-1*torch.ones(n,device=device,dtype=torch.int64)
    cost=torch.mul(Lambda,n)
    cost_pre=cost
    L_pre=L.clone()
    return cost,L,cost_pre,L_pre

@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:],nb.float32,nb.int64[:]))(nb.float32[:,:],nb.int64,nb.int64,nb.float32)],nopython=True)
def one_x_opt(M1,i_act:nb.int64,j_act:nb.int64,Lambda:nb.float32): 
    '''
    Parameters
    ----------
    M1 : 1*m np float array, if M1 has 2 ore more rows or M1 is empty array, there is a mistake in the main loop.  
    j_act : integer>=0 or =-1
    Lambda : float number>=0
    
    Returns
    -------
    cost: float number
        cost for the one point X opt problem. 
    list: np 1*1 array, whose entries could be 0,1,... or -1
        transportation plan, which contains only one element, 
    cost_pre
        cost for the previous problem of the one point X opt problem 
    list_pre: np 1*1 array, whose entries could be 0,1,2.... or -1
        transportation plan 
    
    
    if j_act>=0 and M[0,j_act]<Lambda:
        return:
            cost=M[0,j_act]
            L=[j_act]
            cost_pre=0
            L_pre=[]
    In other case, return:
        cost=Lambda
        L=[-1]
        cost_pre=Lambda
        L_pre=[-1]
    Ex. M1=[0,1.0,4.0], j_act=-1, Lambda=0.4 
    return 0.4,[-1],0.4,[-1]
    Ex. M1=[0,1.0,4.0], j_act=1, Lambda=0.4
    return 0.4 [-1], 0.4,[-1]
    Ex. M1=[0,1.0,4.0], j_act=1, Lambda=2
    return 1.0, [1], 0,[]
    '''       
    if j_act<0:
        return Lambda,np.array([-1],dtype=np.int64),Lambda,np.array([-1],dtype=np.int64)
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,np.array([-1],dtype=np.int64),Lambda,np.array([-1],dtype=np.int64)
    else:
        return c_xy,np.array([j_act],dtype=np.int64),np.float32(0),np.empty(0,dtype=np.int64)

@nb.njit()
def merge_list(L):
    n=len(L) 
    merged_array=L[0]
    for i in range(1,n):
        merged_array=np.concatenate((merged_array,L[i]))
    return merged_array
  
    

@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:],nb.float32,nb.int64[:]))(nb.float32[:,:],nb.int64,nb.int64,nb.float32)],nopython=True)
def one_x_opt_np(M1,i_act:nb.int64,j_act:nb.int64,Lambda:nb.float32): 
    '''
    Parameters
    ----------
    M1 : 1*m np float array, if M1 has 2 ore more rows or M1 is empty array, there is a mistake in the main loop.  
    j_act : integer>=0 or =-1
    Lambda : float number>=0
    Returns
    -------
    cost: float number
        cost for the one point X opt problem. 
    list: np 1*1 array, whose entries could be 0,1,... or -1
        transportation plan, which contains only one element, 
    cost_pre
        cost for the previous problem of the one point X opt problem 
    list_pre: np 1*1 array, whose entries could be 0,1,2.... or -1
        transportation plan 
    
    if j_act>=0 and M[0,j_act]<Lambda:
        return:
            cost=M[0,j_act]
            L=[j_act]
            cost_pre=0
            L_pre=[]
    In other case, return:
        cost=Lambda
        L=[-1]
        cost_pre=Lambda
        L_pre=[-1]
    Ex. M1=[0,1.0,4.0], j_act=-1, Lambda=0.4 
    return 0.4,[-1],0.4,[-1]
    Ex. M1=[0,1.0,4.0], j_act=1, Lambda=0.4
    return 0.4 [-1], 0.4,[-1]
    Ex. M1=[0,1.0,4.0], j_act=1, Lambda=2
    return 1.0, [1], 0,[]
    '''       
    if j_act<0:
        return Lambda,np.array([-1],dtype=np.int64),Lambda,np.array([-1],dtype=np.int64)
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,np.array([-1],dtype=np.int64),Lambda,np.array([-1],dtype=np.int64)
    else:
        return c_xy,np.array([j_act],dtype=np.int64),np.float32(0),np.empty(0,dtype=np.int64)
    
        
@torch.jit.script  
def one_x_opt_T(M1: torch.Tensor,i_act:int,j_act:torch.Tensor,Lambda: torch.Tensor)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    '''
    Parameters
    ----------
    M1 : 1*m np float array, if M1 has 2 ore more rows or M1 is empty array, there is a mistake in the main loop.  
    j_act : integer>=0 or =-1
    Lambda : float number>=0
    Returns
    -------
    cost: float number
        cost for the one point X opt problem. 
    list: np 1*1 array, whose entries could be 0,1,... or -1
        transportation plan, which contains only one element, 
    cost_pre
        cost for the previous problem of the one point X opt problem 
    list_pre: np 1*1 array, whose entries could be 0,1,2.... or -1
        transportation plan 
    
    if j_act>=0 and M[0,j_act]<Lambda:
        return:
            cost=M[0,j_act]
            L=[j_act]
            cost_pre=0
            L_pre=[]
    In other case, return:
        cost=Lambda
        L=[-1]
        cost_pre=Lambda
        L_pre=[-1]
    Ex. M1=[0,1.0,4.0], j_act=-1, Lambda=0.4 
    return 0.4,[-1],0.4,[-1]
    Ex. M1=[0,1.0,4.0], j_act=1, Lambda=0.4
    return 0.4 [-1], 0.4,[-1]
    Ex. M1=[0,1.0,4.0], j_act=1, Lambda=2
    return 1.0, [1], 0,[]
    '''      
    device=M1.device.type
    if j_act<0:
        return Lambda,torch.tensor([-1],device=device,dtype=torch.int64),Lambda,torch.tensor([-1],device=device,dtype=torch.int64)
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,torch.tensor([-1],device=device,dtype=torch.int64),Lambda,torch.tensor([-1],device=device,dtype=torch.int64)
    else:
        return c_xy,j_act.reshape(1),torch.tensor(0,device=device,dtype=torch.float32),torch.empty(0,device=device,dtype=torch.int64)



    

    

@nb.jit([nb.float32[:](nb.float32[:,:],nb.int64[:],nb.int64[:])],nopython=True)
def matrix_take(X,L1,L2):
    return np.array([X[L1[i],L2[i]] for i in range(L1.shape[0])])