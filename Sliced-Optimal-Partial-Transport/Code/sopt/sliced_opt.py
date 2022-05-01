# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:58:08 2022

@author: laoba
"""

import numpy as np
import math
from library import *
from opt import *
import ot
import torch
from ot.sliced import get_random_projections
from torch import optim
import multiprocessing as mp
#import timebudget

def run_complex_operations(operation, input, pool):
    pool.map(operation, input)


def random_projections(d,n_projections,device='cuda',dtype=torch.float):
    Gaussian_vector=torch.normal(0,1,size=[d,n_projections],device='cuda',dtype=dtype)
    projections=Gaussian_vector/torch.sqrt(torch.sum(torch.square(Gaussian_vector),0))    
    return projections

    

def sopt(X,Y,Lambda=40,n_projections=300):
    '''
    X: X is n*d dimension torch array
    Y: Y is m*d dimension torch array
    '''
    d=X.shape[1]
    device=X.device.type
    dtype=X.dtype
    projections=random_projections(d,n_projections,device,dtype)
    X_sliced=torch.matmul(projections.T,X.T)
    Y_sliced=torch.matmul(projections.T,Y.T)
    cost_sum=0
    for i in range(0,n_projections):
        X_theta=X_sliced[i,:]
        Y_theta=Y_sliced[i,:]
        X_theta=X_theta.sort().values
        Y_theta=Y_theta.sort().values
        cost,L=opt_1d_v2(X_theta, Y_theta, Lambda)
        cost_trans=refined_cost(X_theta,Y_theta,L,Lambda,penulty=True)
        cost_sum+=cost_trans
    cost_sum=cost_sum/n_projections
    return cost_sum



class sopt_pool():
    '''
    X: X is n*d dimension torch array
    Y: Y is m*d dimension torch array
    '''
   
    def D1(self,i):
        X_theta=X_sliced[i,:]
        Y_theta=Y_sliced[i,:]
        X_theta=X_theta.sort().values
        Y_theta=Y_theta.sort().values
        cost,L=opt_1d_v2(X_theta, Y_theta, Lambda)
        cost_trans=refined_cost(X_theta,Y_theta,L,Lambda,penulty=True)
        cost_sum[i]=cost_trans
    def run(self,operation,input,pool):
        pool.map(operation,input)
    
    def __init__(X,Y,Lambda=40,n_projections=100):
        d=X.shape[1]
        device=X.device.type
        dtype=X.dtype
        projections=random_projections(d,n_projections,device,dtype)
        X_sliced=torch.matmul(projections.T,X.T)
        Y_sliced=torch.matmul(projections.T,Y.T)
        cost_all=torch.tensor([n_projections])
        process_count=10
        process_pool=mp.Pool(process_count)
        
        run(D1,range(10),process_pool)

        cost_sum=torch.sum(cost_all)/n_projections
        return cost_sum




def sopt_es(X,Y,Lambda=40,n_projections=50):
    '''
    X: X is n*d dimension array
    Y: Y is m*d dimension array
    '''
    device=X.device.type
    dtype=X.dtype
    d=X.shape[1]
    projections=random_projections(d,n_projections,device,dtype)   
    X_sliced=torch.matmul(projections.T,X.T)
    Y_sliced=torch.matmul(projections.T,Y.T)
    cost_sum=0     
    n=X.shape[0]
    m=Y.shape[0]
    M=cost_matrix_T(X,Y)
    for i in range(0,n_projections):
        X_theta=X_sliced[i,:]
        Y_theta=Y_sliced[i,:]
        X_theta_s=X_theta.sort().values
        Y_theta_s=Y_theta.sort().values
        indices_X=X_theta.sort().indices
        indices_Y=Y_theta.sort().indices  
        cost,L=opt_1d_np(X_theta_s, Y_theta_s, Lambda)
        L=recover_indice(indices_X,indices_Y,L,device)
        plan=list_to_plan(L, m,device)
        cost=torch.sum(M*plan)
        cost_sum+=cost
    return cost_sum/n_projections





def max_sopt(X,Y,Lambda=40,n_projections=50):
    '''
    X: n*d dimension torch array
    Y: m*d dimension torch array
    '''
    device=X.device.type
    dtype=X.dtype
    d=X.shape[1]
    projections=random_projections(d,n_projections)
    projections=projections.to(device=device).to(dtype=dtype)
    X_sliced=torch.matmul(projections.T,X.T)
    Y_sliced=torch.matmul(projections.T,Y.T)
    cost_opt=-1
    for i in range(0,n_projections):
        print(i)
        X_theta=X_sliced[i,:]
        Y_theta=Y_sliced[i,:]
        X_theta=X_theta.sort().values
        Y_theta=Y_theta.sort().values
        cost,L=opt_1d_v2(X_theta, Y_theta, Lambda)
        if cost>cost_opt:
            L_opt=L
            X_opt=X_theta
            Y_opt=Y_theta
    cost_opt=refined_cost(X_opt,Y_opt,L_opt,Lambda,penulty=False)
    return cost_opt



# from timebudget import timebudget
# from multiprocessing import Pool

# iterations_count = round(1e7)

# def complex_operation(input_index):
#     print("Complex operation. Input index: {:2d}\n".format(input_index))
#     [math.exp(i) * math.sinh(i) for i in [1] * iterations_count]


# def run_complex_operations(operation, input, pool):
#     pool.map(operation, input)

# processes_count = 10

# if __name__ == '__main__':
#     processes_pool = Pool(processes_count)
#     run_complex_operations(complex_operation, range(10), processes_pool)   



