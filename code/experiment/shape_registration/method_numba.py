

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:01:32 2022

@author: baly
"""



import sys

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
from sopt.library import *
from sopt.lib_shape import *
from sopt.sliced_opt import *   

# our method
@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64,int64)'])
def sopt_main(T,S,n_iterations,N0):
    n,d=T.shape
    N1=S.shape[0]
    # initlize 

    rotation=np.eye(d,dtype=np.float32) #.astype(Dtype)    
    scalar=np.float32(1) #
    beta=vec_mean(T)-vec_mean(scalar*np.dot(S,rotation)) 
    #paramlist=[]
    projections=random_projections_nb(d,n_iterations,1)
    mass_diff=0
    b=np.log((N1-N0+1)/1)
    Lambda=3*np.sum(beta**2)
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    
    Lx_hat_org=arange(0,n)
    Delta=Lambda/8
    lower_bound=Lambda/100
    for i in range(n_iterations):
        theta=projections[i]
        T_hat_theta=np.dot(theta,T_hat.T)
        T_theta=np.dot(theta,T.T)
        
        T_hat_indice=T_hat_theta.argsort()
        T_indice=T_theta.argsort()
        T_hat_s=T_hat_theta[T_hat_indice]
        T_s=T_theta[T_indice]
        cost,L=opt_1d_v2_a(T_hat_s,T_s,Lambda)
        L=recover_indice(T_hat_indice,T_indice,L)
        
        #move That
        Lx_hat=Lx_hat_org.copy()
        Lx_hat=Lx_hat[L>=0]
        mass=Lx_hat.shape[0]
        if Lx_hat.shape[0]>=1:
            Lx=L[L>=0]
            T_hat_take=T_hat_theta[Lx_hat]
            T_take=T_theta[Lx]
            T_hat[Lx_hat]+=(T_take-T_hat_take).reshape(-1,1)*theta
        
        T_hat_take=T_hat[Lx_hat]
        S_take=S[Lx_hat]
        
        rotation,scalar_d=recover_rotation_nb(T_hat_take,S_take)
        scalar=np.float32(np.sqrt(np.trace(np.cov(T_hat_take.T))/np.trace(np.cov(S_take.T))))
        beta=vec_mean(T_hat_take)-vec_mean(scalar*S_take.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
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
        

    return rotation_list,scalar_list,beta_list    

# # method of spot_boneel method
@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64,int64)'])
def spot_bonneel(T,S,n_projections=20,n_iterations=200):
    n,d=T.shape
    N1=S.shape[0]
    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    #paramlist=[]
    
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
        projections=random_projections_nb(d,n_projections,1)
        X_correspondence_pot(T_hat,T,projections)
        
        rotation,scalar=recover_rotation_nb(T_hat,S)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        #move That         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list    


@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64)'])
def icp_du(T,S,n_iterations):
    n,d=T.shape

    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1) #
    beta=vec_mean(T)-vec_mean(scalar*np.dot(S,rotation))
    #paramlist=[]
    
    
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    T_hat=np.dot(S,rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar_d=recover_rotation_du_nb(T_hat,S)
        scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  



@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64)'])
def icp_umeyama(T,S,n_iterations):
    n,d=T.shape

    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    # paramlist=[]
    rotation_list=np.zeros((n_iterations,d,d)).astype(np.float32)
    scalar_list=np.zeros((n_iterations)).astype(np.float32)
    beta_list=np.zeros((n_iterations,d)).astype(np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    

    
    for i in range(n_iterations):
#        print(i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar=recover_rotation_nb(T_hat,S)
        #scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        X_hat=S.dot(rotation)*scalar+beta
        
        #move That         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  


# item_list=['/stanford_bunny',]
# exp_num=item

# label_L=['0','1','2','3']
# L=['/10k','/9k','/8k','/7k']
# time_list={}
# for (label,per_s) in [('0','-7p'),('1','-5p')]:
#     n_point=L[int(label)]    
#     data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
#     save_path='experiment/shape_registration/result'+exp_num+n_point
#     data=torch.load(data_path+item+'.pt')
    
#     time_dict={}
    
#     T0=data['T0'].to(torch.float32)
#     S0=data['S0'+label].to(torch.float32)
#     T1=data['T1'+per_s].to(torch.float32)
#     S1=data['S1'+label+per_s].to(torch.float32)
    
#     T=T1.numpy().copy()
#     S=S1.numpy().copy()
#     N0=S0.shape[0]
#     start_time=time.time()
#     n_iterations=4000
#     sopt_main(T,S,n_iterations,N0)
#     end_time=time.time()
#     wall_time=end_time-start_time

#     result={}
#     result['wall_time']=wall_time
#     result['n_iterations']=n_iterations
#     result['per_time']=wall_time/n_iterations
#     time_dict['sopt']=result
    

    
#     T=T1.numpy().copy()
#     S=S1.numpy().copy()
#     n_projections=20
#     n_iterations=200
#     start_time=time.time()
#     spot_bonneel(T,S,n_projections,n_iterations)
#     end_time=time.time()
#     wall_time=end_time-start_time
#     result={}
#     result['wall_time']=wall_time
#     result['n_iterations']=n_iterations
#     result['per_time']=wall_time/n_iterations
#     time_dict['spot']=result
    

    

    
#     T=X1.numpy().copy()
#     S=Y1.numpy().copy()
#     n_iterations=400
    
#     start_time=time.time()
#     icp_du(T,S,n_iterations)
#     end_time=time.time()
#     wall_time=end_time-start_time
#     result={}
#     result['wall_time']=wall_time
#     result['n_iterations']=n_iterations
#     result['per_time']=wall_time/n_iterations
#     time_dict['icp-du']=result  
    
#     T=T1.numpy().copy()
#     S=S1.numpy().copy()
#     n_iterations=400
    
#     start_time=time.time()
#     icp_umeyama(T,S,n_iterations)
#     end_time=time.time()
#     wall_time=end_time-start_time
#     result={}
#     result['wall_time']=wall_time
#     result['n_iterations']=n_iterations
#     result['per_time']=wall_time/n_iterations
#     time_dict['icp-umeyama']=result 
    


#     time_list[n_point+'-'+per_s]=time_dict
    


# torch.save(time_list,'experiment/shape_registration/result/'+str(item)+'-time_list.pt')



    
    
    
    
