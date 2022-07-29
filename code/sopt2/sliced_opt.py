# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:58:08 2022

@author: laoba
"""
import os
import numpy as np
from typing import Tuple
import torch
from scipy.stats import ortho_group
import sys
import numba as nb
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
from sopt2.library import *
from sopt2.opt import *




def random_projections(d,n_projections,device='cpu',dtype=torch.float,Type=None):
    '''
    input: 
    d: int 
    n_projections: int

    output: 
    projections: d*n torch tensor

    '''
    if Type==None:
#        torch.manual_seed(0)
        Gaussian_vector=torch.normal(0,1,size=[d,n_projections],device=device,dtype=dtype)
        projections=Gaussian_vector/torch.sqrt(torch.sum(torch.square(Gaussian_vector),0))
        projections=projections.T
    elif Type=='orth':
 #       np.random.seed(0)
        r=int(n_projections/d)
        projections=np.concatenate([ortho_group.rvs(d) for i in range(r)],axis=1)
        projections=torch.from_numpy(projections).to(device=device).to(dtype=dtype).T
    else:
        print('Type must be None or orth')
    return projections

@nb.njit([nb.int64[:,:](nb.float32[:,:],nb.float32[:,:],nb.float32)],parallel=True,fastmath=True)
def allplans_s(X_sliced,Y_sliced,Lambda):
    N,n=X_sliced.shape
    plans=np.zeros((N,n),dtype=np.int64)
    for i in nb.prange(N):
        X_theta=X_sliced[i]
        Y_theta=Y_sliced[i]
        M=cost_matrix(X_theta,Y_theta)
        cost,L=opt_1d_v2_apro(X_theta,Y_theta,Lambda)
        plans[i]=L
    return plans


@nb.njit([nb.types.Tuple((nb.int64[:],nb.float32))(nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32,nb.int64)])
def X_correspondence(X,Y,projections,Lambda,mass):
    N,d=projections.shape
    n=X.shape[0]
    Delta=Lambda*1/10
    Lx_org=arange(0,n)
    frequency=np.zeros(n,dtype=np.int64)
    for i in range(N):
        theta=projections[i]
        X_theta=mat_vec_mul(X.T,theta)
        Y_theta=mat_vec_mul(Y.T,theta)
        X_indice=X_theta.argsort()
        Y_indice=Y_theta.argsort()
        X_s=X_theta[X_indice]
        Y_s=Y_theta[Y_indice]
        cost,L=pot_1d(X_s,Y_s)
#        cost,L=opt_1d_v2_apro(X_s,Y_s,Lambda)
        L=recover_indice(X_indice,Y_indice,L)
        #move X
        Lx=Lx_org.copy()
        Lx=Lx[L>=0]
        if Lx.shape[0]>=1:
            Ly=L[L>=0]
            X_take=X_theta[Lx]
            Y_take=Y_theta[Ly]
            X[Lx]+=transpose(Y_take-X_take)*theta
            frequency[Lx]+=1       
          

        #update Lambda
        mass1=np.sum(L>=0)
        mass_diff=mass1-mass
        if mass_diff>mass*0.02:
            Lambda-=Delta
        if mass_diff<-mass*0.02:
            Lambda+=Delta
        if Lambda<=Delta:
            Lambda=Delta
            Delta=Delta/2
    print('mass_diff',mass_diff)

    return frequency,Lambda
    
    
    
class sopt():    
    def __init__(self,X,Y,Lambda,n_projections,Type=None):
        self.X=X
        self.Y=Y
        self.device=X.device.type
        self.dtype=X.dtype
        self.n,self.d=X.shape
        self.m=Y.shape[0]
        self.n_projections=n_projections
        self.Lambda=Lambda
        self.Type=Type


    def sliced_cost(self,penulty=False):
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.plans,penulty)
        mass=torch.sum(self.plans>=0)/self.plans.shape[0]
        return cost,mass
    
    def get_directions(self):
        self.projections=random_projections(self.d,self.n_projections,self.device,self.dtype)

    def get_all_projections(self):
        self.X_sliced=torch.matmul(self.projections,self.X.T)
        self.Y_sliced=torch.matmul(self.projections,self.Y.T)
        
    def get_one_projection(self,i):
        self.X_sliced=torch.matmul(self.projections[i],self.X.T).unsqueeze(0)
        self.Y_sliced=torch.matmul(self.projections[i],self.Y.T).unsqueeze(0)
    

    

    def get_plans(self):
        X_sliced_s,indices_X=self.X_sliced.detach().sort()
        Y_sliced_s,indices_Y=self.Y_sliced.detach().sort()
        #Lambda=np.float32(self.Lambda)
        X_sliced_np=X_sliced_s.cpu().numpy()
        Y_sliced_np=Y_sliced_s.cpu().numpy()
        plans=allplans_s(X_sliced_np,Y_sliced_np,self.Lambda)
        plans=torch.from_numpy(plans).to(device=self.device,dtype=torch.int64)
        self.plans=recover_indice_M(indices_X,indices_Y,plans)

    
    def refined_cost(self,Xs,Ys,plans,penulty=True):
        N=Xs.shape[0]
        self.Lx=[torch.arange(self.n,device=self.device)[plans[i]>=0] for i in range(N)]
        self.Ly=[plans[i][plans[i]>=0] for i in range(N)]
        self.X_take=torch.cat([Xs[i][self.Lx[i]] for i in range(N)])
        self.Y_take=torch.cat([Ys[i][self.Ly[i]] for i in range(N)])        
        cost_trans=torch.sum(cost_function_T(self.X_take, self.Y_take))
        destroy_mass=N*self.n-self.X_take.shape[0]
        penulty_value=self.Lambda*destroy_mass
        if penulty==True:
            return (cost_trans+penulty_value)/N    
        elif penulty==False:
            return cost_trans/N
        

class sopt_correspondence(sopt):
    def __init__(self,X,Y,Lambda,N_projections=20,Type=None):
        sopt.__init__(self,X,Y,Lambda,N_projections,Type)
    def correspond(self,mass):
        self.X_frequency,self.Lambda=X_correspondence(self.X.numpy(),self.Y.numpy(),self.projections.numpy(),self.Lambda,mass)
        X_order=self.X_frequency.argsort()
        Lx=arange(0,self.n)
#        self.Lx=torch.from_numpy(Lx[self.X_frequency>=18])
        self.Lx=torch.from_numpy(Lx[X_order>=self.n-mass]) 
#        self.X_take=self.X[self.Lx]
        
#        self.X[Lx]=self.X_trans[Lx] # we only trans the x which has highest frequency to move
        
        
        
    


        
        
   
        
        

    
        
    



        

class max_sopt(sopt):
    
    def max_cost(self):
        max_index=self.costs.argmax()
        max_plan=self.plans[max_index].reshape([1,self.n])
        X_max=self.X_sliced[max_index].reshape([1,self.n])
        Y_max=self.Y_sliced[max_index].reshape([1,self.m])
        max_cost=self.refined_cost(X_max, Y_max, max_plan)
        max_mass=torch.sum(max_plan>=0)        
        return max_cost,max_mass
    

# class sopt_majority(sopt):
#     def __init__(self,X,Y,Lambda,n_projections=2,Type=None,n_destroy=0):
#         sopt_for.__init__(self,X,Y,Lambda,n_projections,Type)
#         #self.n_preserve=N
#         self.new_plan(n_destroy)
    
#     def new_plan(self,n_destroy):
#         self.new_plans=self.plans.clone()
#         X_frequency=torch.sum(self.plans>=0,0)
#         lowest_frequency=X_frequency.sort().indices[0:n_destroy]
#         self.new_plans[:,lowest_frequency]=-1
#     def sliced_cost(self):
#         cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.new_plans)
#         mass=torch.sum(self.plans>=0)/self.n_projections
#         return cost,mass

class sopt_majority_cut(sopt):
    def __init__(self,X,Y,Lambda,n_projections=2,Type=None,cut=0):
        sopt_for.__init__(self,X,Y,Lambda,n_projections,Type)
        self.new_plan(cut)
    
    def new_plan(self,cut):
        self.new_plans=self.plans.clone()
        X_frequency=torch.sum(self.plans>=0,0)
        
        self.new_plans[:,X_frequency<=cut]=-1
    def sliced_cost(self):
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.new_plans)
        mass=torch.sum(self.plans>=0)/self.n_projections
        return cost,mass
    
    
        

        
    

    
    
    
    

        
        

    

    


def sopt_orig(X,Y,Lambda=40,n_projections=50,Type=None):
    '''
    input: 
    X: X is n*d dimension torch tensor
    Y: Y is m*d dimension torch tensor

    output: 
    cost_sum: 0-dimension tensor
    n_point_sum: int tensor
    '''
    d=X.shape[1]
    device=X.device.type
    dtype=X.dtype
    projections=random_projections(d,n_projections,device,dtype,Type)
    X_sliced=torch.matmul(projections,X.T)
    Y_sliced=torch.matmul(projections,Y.T)
    cost_sum=0
    n_point_sum=0
    for i in range(0,n_projections):
        X_theta=X_sliced[i,:]
        Y_theta=Y_sliced[i,:]
        X_theta1=X_theta.sort().values
        Y_theta1=Y_theta.sort().values
        cost,L=opt_1d_T(X_theta1, Y_theta1, Lambda)
        cost_sum+=cost
        n_point_sum+=torch.sum(L>=0)

        
    cost_sum=cost_sum/n_projections
    n_point_sum=n_point_sum/n_projections
    return cost_sum,n_point_sum



def max_sopt_orig(X,Y,Lambda=40,n_projections=50,Type=None):
    '''
    input: 
    X: X is n*d dimension torch tensor
    Y: Y is m*d dimension torch tensor

    output: 
    cost_sum: 0-dimension tensor
    n_point_sum: int tensor
    '''
    device=X.device.type
    dtype=X.dtype
    d=X.shape[1]
    projections=random_projections(d,n_projections,device,dtype,Type)
    X_sliced=torch.matmul(projections,X.T)
    Y_sliced=torch.matmul(projections,Y.T)
    cost_opt=-1
    for i in range(0,n_projections):
        X_theta=X_sliced[i,:]
        Y_theta=Y_sliced[i,:]
        X_theta=X_theta.sort().values
        Y_theta=Y_theta.sort().values
        cost,L=opt_1d_T(X_theta, Y_theta, Lambda)
        if cost_opt<cost:
            cost_opt=cost
            L_opt=L
    mass_opt=torch.sum(L_opt>=0)

    return cost_opt,mass_opt


    
    
    
    
        
    

    
    




    


