#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:47:09 2022

@author: baly
"""
import sys
import open3d as o3d
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import ot

work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)

method='/icp_umeyama'
data_list=[
'/witchcastle',
'/mumble_sitting',
'/dragon', 
'/mumble_sitting']

#extract the result
experiment='/10k-5p'
result_path=parent_path+'/experiment/shape_registration/result'



#extract the grand truth
data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
param_op_list=[]
trans_list=[]
trans_op_list=[]
for data in data_list:
    # extract the data 
    data_pt_path=data_path+data+'.pt'
    data_pt=torch.load(data_pt_path)
    param_op=data_pt['param']
    param_op_list.append(param_op)
    rotation_op=param_op['rotation_op']
    scalar_op=param_op['scalar_op']
    n=data_pt['X0'].shape[0] # nomalized 
    std=torch.sqrt(torch.trace(torch.cov(data_pt['X0'].T)*(n-1)/n))
    beta_op=param_op['beta_op']/std
    trans_op=torch.cat((beta_op.reshape((3,1)),rotation_op*scalar_op),1)
    trans_op_list.append(trans_op)
    
    #extradt the parameter
    path=result_path+data+experiment+method+'_param.pt'
    param=torch.load(path)[-1]
    rotation=param['rotation']
    scalar=param['scalar']
    beta=param['beta']/std
    trans=torch.cat((beta.reshape((3,1)),rotation*scalar),1)
    trans_list.append(trans)
    
#trans_op_list=[]
#for param_op in param_op_list:
#    rotation_op=param_op['rotation_op']
#    scalar_op=param_op['scalar_op']
#    beta_op=param_op['beta_op']
#    trans_op=torch.cat((beta_op.reshape((3,1)),rotation_op*scalar_op),1)
#    trans_op_list.append(trans_op)

n=len(trans_op_list)
error_list=torch.zeros(n)
for i in range(n):
    trans=trans_list[i]
    trans_op=trans_op_list[i]
    err=torch.norm(trans.reshape(-1)-trans_op.reshape(-1))
    error_list[i]=err

median=torch.median(error_list) 
variance=torch.var(error_list)
print('median of error', median)
print('variance of error', variance)

