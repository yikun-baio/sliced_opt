# # -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:32:17 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu 
"""

import numpy as np
import torch
import os

import numba as nb
from typing import Tuple #,List
from numba.typed import List


# def text():
#     print('here')

# print('here')

@nb.njit(cache=True)
def argmin_nb(X,Y):
    Min=np.inf
    ind=0
    m=Y.shape[0]
    for i in range(m):
        cost_xy=X[i]-Y[i]
        if cost_xy<Min:
            Min=cost_xy
            ind=i
    return ind


@nb.njit(cache=True)
def cost_function(x,y,p=2): 
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
    V=np.abs(x-y)**p
    return V



# @nb.njit(['float64[:,:](float64[:])'],fastmath=True)
# def transpose(X):
#     n=X.shape[0]
#     XT=np.zeros((n,1),np.float64)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT

# @nb.njit(['float32[:,:](float32[:])'],fastmath=True)
# def transpose_32(X):
#     n=X.shape[0]
#     XT=np.zeros((n,1),np.float32)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT


@nb.njit(['float64[:,:](float64[:],float64[:])','float32[:,:](float32[:],float32[:])'],fastmath=True,cache=True)
def cost_matrix(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
    XT=np.expand_dims(X,1)
    M=cost_function(XT,Y)    
    return M


@nb.njit(cache=True)
def argmin_nb(array):
    Min=np.inf
    Min_ind=0
    n=array.shape[0]
    for i in range(n):
        val=array[i]
        if val<Min:
            Min=val
            Min_ind=i
    return Min_ind,Min






#@nb.njit(fastmath=True)
@nb.njit(['float32[:,:](float32[:,:],float32[:,:])','float64[:,:](float64[:,:],float64[:,:])'],fastmath=True,cache=True)
def cost_matrix_d(X,Y):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
#    n,d=X.shape
#    m=Y.shape[0]
#    M=np.zeros((n,m)) 
    # for i in range(d):
    #     C=cost_function(X[:,i:i+1],Y[:,i])
    #     M+=C
    X1=np.expand_dims(X,1)
    Y1=np.expand_dims(Y,0)
    M=np.sum(cost_function(X1,Y1),2)
    return M




@nb.njit(['Tuple((int64,float32))(float32,float32[:])','Tuple((int64,float64))(float64,float64[:])'],fastmath=True,cache=True) 
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


@nb.njit(['int64[:](float64[:,:])','int64[:](float32[:,:])'],fastmath=True,cache=True)
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
    argmin_Y=np.zeros(n,np.int64)
    for i in range(n):
        argmin_Y[i]=M[i,:].argmin()
    return argmin_Y



@nb.njit(['int64[:],int64'],cache=True) 
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

         

@nb.njit(['Tuple((int64,int64))(int64[:])'],cache=True)  
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


@nb.njit(['Tuple((int64,int64))(int64[:])'],cache=True)
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

@nb.njit(['int64[:](int64,int64)'],fastmath=True,cache=True)
def arange(start,end):
    n=end-start
    L=np.zeros(n,np.int64)
    for i in range(n):
        L[i]=i+start
    return L



@nb.njit(['Tuple((int64,int64))(int64[:])'],cache=True)
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



@nb.njit(['Tuple((int64,int64))(int64[:])'],cache=True)
def unassign_y_nb(L1):
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
    
    j_last=L1[-1]
    n=L1.shape[0]
    L_range=arange(j_last-n+1,j_last+1)
    L_dif=np.where(L_range-L1>0)[0]
    if L_dif.shape[0]==0:
        return 0, L1[0]-1
    else:
        i_act=L_dif[-1]+1
        j_act=L_range[i_act-1]
    return i_act,j_act



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



@nb.njit(['int64[:](int64[:],int64[:],int64[:])'],cache=True)
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


@nb.njit(['int64[:,:](int64[:],int64)'],fastmath=True,cache=True)
def array_to_matrix(L,m):
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
    plan=np.zeros((n,m),np.int64)
    
    Ly=L[L>=0]
    Lx=arange(0,n)
    Lx=Lx[L>=0]
    for i in Lx:
        plan[i,L[i]]=1
    return plan

@nb.njit(['int64[:](int64[:,:])'],fastmath=True,cache=True)
#@nb.njit(fastmath=True)
def L_to_pi(L_lp):
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
    n,m=L_lp.shape
    L=np.full(n,-1,np.int64)
    for i in range(n):
        indexes=np.where(L_lp[i,:]>=0.5)[0]
        if indexes.shape[0]==1:
            L[i]=indexes[0]
        elif indexes.shape[0]>=2:
            print('error')
    return L


# @nb.njit(['Tuple((float64,int64[:],float64,int64[:]))(int64,float64)'])
# def empty_Y_opt(n,Lambda):
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
#     L : n*1 list. 
#         transportation plan, whose entry could be 0,1,... or -1. 
#         must be in increasing order
#         in this function, it can only contains -1 
#     cost_pre : float number 
#         cost of the opt problem where Y is empty
#         in this example, cost_pre=cost
#     L_pre : n*1 list. 
#         transportation plan, whose entry could be 0,1,... or -1. 
#         must be in increasing order
#         in this function, L_pre=L

#     '''
#     L=np.full(n,-1,dtype=np.int64) #(n,dtype=np.int64)+np.int64(-1)
#     #for i in range(n):
#     #    L[i]=-1
#     cost=Lambda*np.float64(n)
#     cost_pre=cost
#     L_pre=L.copy()
#     return cost,L,cost_pre,L_pre


#
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

@nb.njit(['Tuple((float64,int64[:],float64,int64[:]))(int64,float64)'],cache=True)
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
    cost=Lambda*np.float64(n)
    cost_pre=cost
    L_pre=L.copy()
    return cost,L,cost_pre,L_pre

@nb.njit(['Tuple((float32,int64[:],float32,int64[:]))(int64,float32)'],cache=True)
def empty_Y_opt_32(n,Lambda):
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

@nb.njit(['Tuple((float64,int64[:],float64,int64[:]))(float64[:,:],int64,int64,float64)'],cache=True)
#@nb.njit()
def one_x_opt(M1,i_act,j_act,Lambda): 
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
    dtype=type(Lambda)
    if j_act<0:
        return Lambda,np.array([-1],np.int64),Lambda,np.array([-1],np.int64)
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,np.array([-1],np.int64),Lambda,np.array([-1],np.int64)
    else:
        return c_xy,np.array([j_act],np.int64),np.float64(0),np.empty(0,np.int64)

@nb.njit(fastmath=True,cache=True)
def merge_list(L):
    n=len(L) 
    merged_array=L[0]
    for i in range(1,n):
        merged_array=np.concatenate((merged_array,L[i]))
    return merged_array
  
    

@nb.njit(['Tuple((float64,int64[:],float64,int64[:]))(float64[:,:],int64,int64,float64)'],cache=True)
def one_x_opt_np(M1,i_act,j_act,Lambda): 
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
    dtype=type(Lambda)
    if j_act<0:
        return Lambda,np.array([-1],dtype=np.int64),Lambda,np.array([-1],dtype=np.int64)
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,np.array([-1],dtype=np.int64),Lambda,np.array([-1],dtype=np.int64)
    else:
        return c_xy,np.array([j_act],dtype=np.int64),np.float64(0),np.empty(0,dtype=np.int64)
    
        


    

@nb.njit(['float32[:](float32[:,:],int64[:],int64[:])','float64[:](float64[:,:],int64[:],int64[:])'],fastmath=True,cache=True)
def matrix_take(X,L1,L2):
    return np.array([X[L1[i],L2[i]] for i in range(L1.shape[0])])



@nb.njit(['(float64[:])(float64[:],float64[:],int64)'],fastmath=True,cache=True)
def Gaussian_mixture(mu_list, variance_list,n):
    N=mu_list.shape[0]
    indices=np.random.randint(0,N,n)
    X=np.zeros(n)
    for i in range(n):
        X[i]=np.random.normal(mu_list[indices[i]],variance_list[indices[i]])
    return X

@nb.njit(['(float32[:])(float32[:],float32[:],int64)'],fastmath=True,cache=True)
def Gaussian_mixture_32(mu_list, variance_list,n):
    N=mu_list.shape[0]
    indices=np.random.randint(0,N,n)
    X=np.zeros(n,dtype=np.float32)
    for i in range(n):
        X[i]=np.float32(np.random.normal(mu_list[indices[i]],variance_list[indices[i]]))
    return X


    
    
    
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



    
