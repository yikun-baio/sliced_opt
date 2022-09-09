#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 07:41:12 2022

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
#@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64,nb.int64)])
def sopt_main(X,Y,n_iterations,N0):
    n,d=X.shape
    N1=Y.shape[0]
    
    # initlize
    rotation=torch.eye(3,dtype=torch.float32)
    scalar=1.0 #
    beta=(torch.mean(X1,0)-torch.mean(scalar*Y@rotation,0))
    paramlist=[]
    
    #n_projections=1
    Lambda_list=3*torch.sum(beta**2).reshape(1)
    Delta=Lambda_list[0]*1/8
    X_hat=Y@rotation*scalar+beta   
    
    A=sopt(X_hat,X,Lambda_list,n_iterations,'orth')
    
    mass_diff=0
    b=np.log((N1-N0)/1)

    A.get_directions()
    
    for epoch in range(n_iterations):
        A.get_one_projection(epoch)
        A.get_plans()
        loss,mass=A.sliced_cost()
       
        n=A.X_take.shape[0]
        A.X[A.Lx]+=(A.Y_take-A.X_take).reshape((n,1))*A.projections[epoch]
        #extract the paired data
        Y_take=Y[A.Lx]
        X_hat_take=A.X[A.Lx]
        
        
        rotation,scalar_d=recover_rotation(X_hat_take,Y_take)
        #scalar=torch.mean(scalar_d)
        scalar=torch.sqrt(torch.trace(torch.cov(X_hat_take.T))/torch.trace(torch.cov(Y_take.T)))
        beta=torch.mean(X_hat_take,0)-torch.mean(scalar*Y_take@rotation,0)
        A.X=Y@rotation*scalar+beta
        

        #N=(N1-N0)*np.exp(-epoch/n_iteration*b)+N0
        N=(N1-N0)*1/(1+b*(epoch/n_iterations))+N0

        mass_diff=mass.item()-N    
        if mass_diff>N*0.009:
            A.Lambda_list[0]=A.Lambda_list[0]-Delta 
        if mass_diff<-N*0.003:
            A.Lambda_list[0]=A.Lambda_list[0]+Delta
            Delta=A.Lambda_list*1/8
        if A.Lambda_list[0]<=Delta:
            A.Lambda_list[0]=Delta
            Delta=A.Lambda_list[0]/8 
        param={}
        param['rotation']=rotation
        param['beta']=beta
        param['scalar']=scalar
        paramlist.append(param)
        

    return paramlist    

# method of spot_boneel method
#@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64,nb.int64)])
def spot_bonneel(X,Y,n_projections,n_iterations):
    n,d=X.shape
    N1=Y.shape[0]
    start_time=time.time()

    
    # initlize 
    rotation=torch.eye(3,dtype=torch.float32)
    scalar=1.0 #
    beta=(torch.mean(X,0)-torch.mean(scalar*Y@rotation,0))
    paramlist=[]

    X_hat=Y@rotation*scalar+beta   
    

    A=spot(X_hat,X,n_projections,'orth')
    mass_diff=0
    for epoch in range(n_iterations):
        A.get_directions()
        A.correspond()
        Y_take=Y
        X_hat_take=A.X
        rotation,scalar=recover_rotation(X_hat_take,Y_take)
        beta=torch.mean(X_hat_take,0)-torch.mean(scalar*Y_take@rotation,0)
        A.X=Y@rotation*scalar+beta        
        param={}
        param['rotation']=rotation
        param['beta']=beta
        param['scalar']=scalar
        paramlist.append(param)
  
    return paramlist


#@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64)])

def icp_du(X,Y,n_iterations):
    n,d=X.shape

    rotation=torch.eye(3,dtype=torch.float32)
    scalar=1.0 #
    beta=(torch.mean(X,0)-torch.mean(scalar*Y@rotation,0))

    paramlist=[]
    X_hat=Y@rotation*scalar+beta       
    for i in range(n_iterations):
        M=cost_matrix_T(X_hat,X)
        argmin_X=M.argmin(dim=1)
        X_take=X[argmin_X]
        X_hat_take=X_take
        rotation,scalar_d=recover_rotation_du(X_hat_take,Y)
        scalar=torch.mean(scalar_d)
        beta=torch.mean(X_hat_take,0)-torch.mean(scalar*Y@rotation,0)
        X_hat=Y@rotation*scalar+beta
        
        param={}
        param['rotation']=rotation
        param['beta']=beta
        param['scalar']=scalar
        paramlist.append(param)
    return paramlist



#@nb.njit([nb.types.Tuple((nb.float32[:,:,:],nb.float32[:],nb.float32[:,:]))(nb.float32[:,:],nb.float32[:,:],nb.int64)])
def icp_umeyama(X,Y,n_iterations):
    n,d=X.shape

    # initlize 
    rotation=torch.eye(d,dtype=torch.float32)
    scalar=1.0 #
    beta=(torch.mean(X,0)-torch.mean(scalar*Y@rotation,0))
    paramlist=[]
    X_hat=Y@rotation*scalar+beta  

    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
        M=cost_matrix_T(X_hat,X)
        argmin_X=M.argmin(dim=1)
        X_take=X[argmin_X]
        X_hat_take=X_take
        rotation,scalar=recover_rotation(X_hat_take,Y)
        #scalar=torch.mean(scalar_d)
        beta=torch.mean(X_hat_take,0)-torch.mean(scalar*Y@rotation,0)
        X_hat=Y@rotation*scalar+beta
        
        param={}
        param['rotation']=rotation
        param['beta']=beta
        param['scalar']=scalar
        paramlist.append(param)
    

    return paramlist


item_list=['/stanford_bunny','/dragon','/mumble_sitting','/witchcastle']

label_L=['0','1','2','3']
L=['/10k','/9k','/8k','/7k']
for item in item_list:
    exp_num=item
    time_list={}
# (label,per_s)=('0','-7p')
# n_point=L[int(label)]    
# data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
# save_path='experiment/shape_registration/result'+exp_num+n_point
# data=torch.load(data_path+item+'.pt')
# time_dict={}
# X0=data['X0'].to(torch.float32)
# Y0=data['Y0'+label].to(torch.float32)
# X1=data['X1'+per_s].to(torch.float32)
# Y1=data['Y1'+label+per_s].to(torch.float32)
# X=X1.clone()
# Y=Y1.clone()
# N0=Y0.shape[0]
# n_projections=100
# n_iterations=200
# start_time=time.time()
# icp_umeyama(X,Y,n_iterations)

    for (label,per_s) in [('0','-7p'),('1','-5p')]:
        n_point=L[int(label)]    
        data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
        save_path='experiment/shape_registration/result'+exp_num+n_point
        data=torch.load(data_path+item+'.pt')
        time_dict={}
        X0=data['X0'].to(torch.float32)
        Y0=data['Y0'+label].to(torch.float32)
        X1=data['X1'+per_s].to(torch.float32)
        Y1=data['Y1'+label+per_s].to(torch.float32)
        X=X1.clone()
        Y=Y1.clone()
        N0=Y0.shape[0]
        start_time=time.time()
        n_iterations=3000
        sopt_main(X,Y,n_iterations,N0)
        end_time=time.time()
        wall_time=end_time-start_time
        
        result={}
        result['wall_time']=wall_time
        result['n_iterations']=n_iterations
        result['per_time']=wall_time/n_iterations
        time_dict['sopt']=result
        
        
        
        X=X1.clone()
        Y=Y1.clone()
        n_projections=100
        n_iterations=200
        start_time=time.time()
        spot_bonneel(X,Y,n_projections,n_iterations)
        end_time=time.time()
        wall_time=end_time-start_time
        result={}
        result['wall_time']=wall_time
        result['n_iterations']=n_iterations
        result['per_time']=wall_time/n_iterations
        time_dict['spot']=result
        
        
        
        X=X1.clone()
        Y=Y1.clone()
        n_iterations=400
        
        start_time=time.time()
        icp_du(X,Y,n_iterations)
        end_time=time.time()
        wall_time=end_time-start_time
        result={}
        result['wall_time']=wall_time
        result['n_iterations']=n_iterations
        result['per_time']=wall_time/n_iterations
        time_dict['icp-du']=result  
        
        X=X1.clone()
        Y=Y1.clone()
        n_iterations=400
        start_time=time.time()
        icp_umeyama(X,Y,n_iterations)
        end_time=time.time()
        wall_time=end_time-start_time
        result={}
        result['wall_time']=wall_time
        result['n_iterations']=n_iterations
        result['per_time']=wall_time/n_iterations
        time_dict['icp-umeyama']=result 
        
        time_list[n_point+'-'+per_s]=time_dict

    torch.save(time_list,'experiment/shape_registration/result'+item+'time_list.pt')