# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:32:17 2022

@author: laoba
"""

import numpy as np
import torch

def cost_function(x,y): 
    V=(x-y)**2
    return V

def cost_function_T(x,y,device='cuda'): 
    return torch.square(x-y)


def cost_matrix(X,Y):
    if len(X.shape)==1:
        X=X.reshape(X.shape[0],1)
        M=cost_function(X,Y)
    elif len(X.shape)==2:
        n=X.shape[0]
        m=Y.shape[0]
        d=X.shape[1]
        M=np.zeros([n,m])
        for i in range(d):
            M+=cost_function(X[:,i:i+1],Y[:,i:i+1].T)        
    return M


def cost_matrix_T(X,Y,device='cuda'):
    if len(X.shape)==1:
        X=X.reshape([X.shape[0],1])
        M=cost_function(X,Y)
    
    elif len(X.shape)==2:
        n,dim=X.shape
        m=Y.shape[0]
        M=torch.zeros([n,m],device=device)
        for d in range(dim):
            M+=cost_function_T(X[:,d:d+1],Y[:,d:d+1].T)
    return M



def closest_y(x,Y):
    cost_list=cost_function(x,Y)    
    min_index=cost_list.argmin()
    min_cost=cost_list[min_index]
    return min_index,min_cost

def closest_y_np(M,k,j_start):
    cost_list=M[k,j_start:]    
    min_index=cost_list.argmin()
    min_cost=cost_list[min_index]
    return min_index,min_cost

def closest_y_T(x,Y):
    cost_list=cost_function_T(x,Y)    
    min_index=cost_list.argmin()
    min_cost=cost_list[min_index]
    return min_index,min_cost


            
def index_adjust(L,j_start=0):
    L=[j+j_start if j>=0 else j for j in L]
    return L


def index_adjust_T(L,j_start=0):
    positive_indices=(L>=0).nonzero()
    L[positive_indices]=L[positive_indices]+j_start
    return None

def index_adjust_np(L,j_start=0):
    positive_indices=(L>=0).nonzero()
    L[positive_indices]=L[positive_indices]+j_start
    return None
         

def startindex(L_pre):    
    i_start=len(L_pre)
    if i_start==0:
        return 0,0
    j_start=max(0,max(L_pre)+1)
    return i_start,j_start

def startindex_T(L_pre):    
    i_start=L_pre.shape[0]
    if i_start==0:
        return 0,0
    j_start=max(0,L_pre.max()+1)
    return i_start,j_start

def list_to_array(*lst):
    """ Convert a list if in numpy format """
    if len(lst) > 1:
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]

def list_to_numpy_array(lst):
    r""" Convert a list if in numpy format """
    return np.array([a.item() for a in lst],dtype=np.float32)



def unassign_y(L1):      
    j_last_assign=L1[-1]
    i_last_assign=len(L1)-1 # this is the value of k-i_start
    for i in range(0,j_last_assign+1):
        j=j_last_assign-i
        i=i_last_assign-i+1
        if j not in L1:
            return i,j

    return 0,-np.inf

def unassign_y_np(L1):    
    j_last=L1[-1]
    n=L1.shape[0]
    L_range=np.arange(j_last-n+1,j_last+1)
    L_dif=np.where(L_range-L1>0)[0]
    if L_dif.shape[0]==0:
        return 0, L1[0]-1
    index_x=L_dif[-1]+1
    index_y=L_range[index_x-1]
    return index_x,index_y


def unassign_y_T(L1,device='cuda'):
    '''
    L1: should not contain negative entry, and is in increasing order
    '''  
    j_last=L1[-1]
    n=L1.shape[0]
    L_range=torch.arange(j_last-n+1,j_last+1,device=device)
    L_dif=torch.where(L_range-L1>0)[0]
    if L_dif.shape[0]==0:
        return 0, L1[0]-1   
    index_x=L_dif[-1]+1
    index_y=L_range[index_x-1]
    return index_x,index_y



def recover_indice(indice_X,indice_Y,L,device='cuda'):
    '''
    indice_X: torch integer tensor
    indice_Y: torch integer tensor 
    L: list
    '''
    n=len(L)
    indice_Y_mapped=torch.tensor([indice_Y[i] if i>=0 else -n-1 for i in L],device=device)
    mapping=torch.stack([indice_X,indice_Y_mapped])
    mapping_final=mapping[:,mapping[0,:].argsort()]
    return mapping_final[1,:]

                     
def refined_cost(X,Y,L,Lambda,penulty=True):
    L_X=[i for i,j in enumerate(L) if j>=0]    
    L_Y=[j for j in L if j>=0]
    Y_take=Y[L_Y]
    X_take=X[L_X]
    num_destruction=len(L)-len(L_X)
    if penulty==True:
        cost=torch.sum(cost_function_T(X_take, Y_take))+2*Lambda*num_destruction
    else:
        cost=torch.sum(cost_function_T(X_take, Y_take))
    return cost

def list_to_plan(L,m,device='cuda'):
    '''
    will return a matrix
    '''
    n=len(L)
    plan=torch.zeros(n,m,device=device)
    for i in range (n):
        if L[i]>=0:
            plan[i,L[i]]=1.0
    return plan
    



def empty_Y_opt(n,Lambda):
    L=[-np.inf]*n
    cost=(Lambda)*n
    return cost,L,cost,L

def empty_Y_opt_np(n,Lambda):
    L=-1*np.ones(n,dtype=int)
    cost=Lambda*n
    return cost,L,cost,L

def empty_Y_opt_T(n,Lambda,device='cuda'):
    L=-1*torch.ones(n,device=device,dtype=int)
    cost=Lambda*n
    return cost,L,cost,L


def one_x_opt(X1,Y1,j_act,Lambda):            
    if j_act<0:
        return Lambda,[-np.inf],Lambda,[-np.inf]
    elif j_act>=0:
        yl=Y1[j_act]
        xk=X1[0]
        c_xkyjk=cost_function(xk,yl)
        if c_xkyjk>=Lambda:
            return Lambda,[-np.inf],Lambda,[-np.inf]
        elif c_xkyjk<Lambda:
            return c_xkyjk,[j_act],0,[]
        
def one_x_opt_np(M1,i_act,j_act,Lambda): 
    if j_act<0:
        return Lambda,np.array([-1]),Lambda,np.array([-1])
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,np.array([-1]),Lambda,np.array([-1])
    else:
        return c_xy,np.array([j_act]),0,np.array([],dtype=int)
    
        

def one_x_opt_T(M1,i_act,j_act,Lambda,device='cuda'):            
    if j_act<0:
        return Lambda,torch.tensor([-1],device=device,dtype=int),Lambda,torch.tensor([-1],device=device,dtype=int)
    c_xy=M1[i_act,j_act]
    if c_xy>=Lambda:
        return Lambda,torch.tensor([-1],device=device,dtype=int),Lambda,torch.tensor([-1],device=device,dtype=int)
    elif c_xy<Lambda:
        return c_xy,j_act.reshape(1),0,torch.empty(0,device=device,dtype=int)


    
    
    




    
