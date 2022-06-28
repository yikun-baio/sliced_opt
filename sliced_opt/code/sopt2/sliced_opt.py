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
import torch.multiprocessing as mp
import sys
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
    elif Type=='orth':
 #       np.random.seed(0)
        r=int(n_projections/d)
        projections=np.concatenate([ortho_group.rvs(d) for i in range(r)],axis=1)
        projections=torch.from_numpy(projections).to(device=device).to(dtype=dtype)
    else:
        print('Type must be None or orth')
    return projections

@nb.jit([nb.int32[:,:](nb.float32[:,:],nb.float32[:,:],nb.float32)],nopython=True,parallel=True)
def allplans(X_sliced,Y_sliced,Lambda):
    N,n=X_sliced.shape
    plans=np.zeros((N,n),dtype=np.int32)
    for i in nb.prange(N):
        X_theta=X_sliced[i]
        Y_theta=Y_sliced[i]
        cost,L=opt_1d_v2(X_theta,Y_theta,Lambda)
        plans[i]=L

    return plans



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
    X_sliced=torch.matmul(projections.T,X.T)
    Y_sliced=torch.matmul(projections.T,Y.T)
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
    X_sliced=torch.matmul(projections.T,X.T)
    Y_sliced=torch.matmul(projections.T,Y.T)
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


    
class sopt_for():
    def __init__(self,X,Y,Lambda,n_projections=6,Type=None):
        self.X=X
        self.Y=Y
        self.device=X.device.type
        self.dtype=X.dtype
        self.n,self.d=X.shape
        self.m=Y.shape[0]
        self.n_projections=n_projections
        self.Lambda=Lambda
        self.Type=Type
        self.get_projections()
        self.get_plans()

    def sliced_cost(self):
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.plans)
        mass=torch.sum(self.plans>=0)/self.n_projections
        return cost,mass

    def get_projections(self):
        projections=random_projections(self.d,self.n_projections,self.device,self.dtype)
        self.X_sliced=torch.matmul(projections.T,self.X.T)
        self.Y_sliced=torch.matmul(projections.T,self.Y.T)
    
    def get_plans(self):
        X_sliced_s,indices_X=self.X_sliced.detach().sort()
        Y_sliced_s,indices_Y=self.Y_sliced.detach().sort()
        Lambda=np.float32(self.Lambda)
        X_sliced_np=X_sliced_s.cpu().numpy()
        Y_sliced_np=Y_sliced_s.cpu().numpy()
        plans=allplans(X_sliced_np,Y_sliced_np,Lambda)
        plans=torch.from_numpy(plans).to(device=self.device,dtype=torch.int64)
        self.plans=recover_indice_M(indices_X,indices_Y,plans)
    
    def refined_cost(self,Xs,Ys,plans):
        N=Xs.shape[0]
        Lx=[torch.arange(self.n,device=self.device)[plans[i]>=0] for i in range(N)]
        Ly=[plans[i][plans[i]>=0] for i in range(N)]
        X_take=torch.cat([Xs[i][Lx[i]] for i in range(N)])
        Y_take=torch.cat([Ys[i][Ly[i]] for i in range(N)])        
        cost_trans=torch.sum(cost_function_T(X_take, Y_take))
        destroy_mass=N*self.n-X_take.shape[0]
        penulty=self.Lambda*destroy_mass
        return (cost_trans+penulty)/N    

# class sopt():
    
#     def __init__(self,X,Y,Lambda,n_projections=6,Type=None):
#         self.device=X.device.type
#         self.dtype=X.dtype
#         self.n,self.d=X.shape
#         self.m=Y.shape[0]
#         self.n_projections=n_projections
#         self.Lambda=Lambda
#         self.Type=Type
#         self.X=X
#         self.Y=Y
#         self.get_projections()
#         self.get_plans()

#     def sliced_cost(self):
#         cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.plans)
#         mass=torch.sum(self.plans>=0)/self.n_projections
#         return cost,mass

#     def get_projections(self):
# #        projections=random_projections(self.d,self.n_projections,self.device,self.dtype)
#         projections=random_projections(self.d,self.n_projections,self.device,self.dtype)
#         self.X_sliced=torch.matmul(projections.T,self.X.T)
#         self.Y_sliced=torch.matmul(projections.T,self.Y.T)
    
#     def get_plans(self):
#         X=self.X
#         Y=self.Y
#         X_sliced=self.X_sliced # remove gradient 
#         Y_sliced=self.Y_sliced # remove gradient 
#         self.X_sliced=self.X_sliced.detach().cpu()
#         self.Y_sliced=self.Y_sliced.detach().cpu()
#         self.X=self.X.detach().cpu()
#         self.Y=self.Y.detach().cpu()
#         self.plans=torch.zeros([self.n_projections,self.n],dtype=torch.int64)
#         self.costs=torch.zeros([self.n_projections])
#         n_cpu=mp.cpu_count()
#         size=int(self.n_projections/n_cpu/2)
#         pool=mp.Pool(6)
#         r=pool.map_async(self.one_slice, range(self.n_projections),chunksize =8) 
#         pool.close()
#         pool.join()
#         self.X_sliced=X_sliced
#         self.Y_sliced=Y_sliced
#         self.X=X
#         self.Y=Y
  
    
    
#     def one_slice(self,i):
#         X_theta=self.X_sliced[i]
#         Y_theta=self.Y_sliced[i]
#         X_theta_s,indices_X=X_theta.sort()
#         Y_theta_s,indices_Y=Y_theta.sort()
#         X_theta_s_np=X_theta_s.numpy()
#         Y_theta_s_np=Y_theta_s.numpy()
#         cost1,L1=opt_1d_np(X_theta_s_np, Y_theta_s_np, self.Lambda)
#         L1=torch.from_numpy(np.int64(L1))
#         L2=recover_indice(indices_X,indices_Y,L1)
#         self.plans[i,:]=L2
#         self.costs[i]=cost1
    
#     def refined_cost(self,Xs,Ys,plans):
#         N=Xs.shape[0]
#         Lx=[torch.arange(self.n)[plans[i]>=0] for i in range(N)]
#         Ly=[plans[i][plans[i]>=0] for i in range(N)]
#         X_take=torch.cat([Xs[i][Lx[i]] for i in range(N)])
#         Y_take=torch.cat([Ys[i][Ly[i]] for i in range(N)])        
#         cost_trans=torch.sum(cost_function_T(X_take, Y_take))
#         destroy_mass=N*self.n-X_take.shape[0]
#         penulty=self.Lambda*destroy_mass
#         return (cost_trans+penulty)/N



        

class max_sopt(sopt_for):
    
    def max_cost(self):
        max_index=self.costs.argmax()
        max_plan=self.plans[max_index].reshape([1,self.n])
        X_max=self.X_sliced[max_index].reshape([1,self.n])
        Y_max=self.Y_sliced[max_index].reshape([1,self.m])
        max_cost=self.refined_cost(X_max, Y_max, max_plan)
        max_mass=torch.sum(max_plan>=0)        
        return max_cost,max_mass
    

class sopt_majority(sopt_for):
    def __init__(self,X,Y,Lambda,n_projections=2,Type=None,n_destroy=0):
        sopt_for.__init__(self,X,Y,Lambda,n_projections,Type)
        #self.n_preserve=N
        self.new_plan(n_destroy)
    
    def new_plan(self,n_destroy):
        self.new_plans=self.plans.clone()
        X_frequency=torch.sum(self.plans>=0,0)
        lowest_frequency=X_frequency.sort().indices[0:n_destroy]
        self.new_plans[:,lowest_frequency]=-1
    def sliced_cost(self):
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.new_plans)
        mass=torch.sum(self.plans>=0)/self.n_projections
        return cost,mass
    
        

        
    

    
    
    
    

        
        

    

    


    
    
    
    
        
    

    
    




    


