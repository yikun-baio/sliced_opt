#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:20:57 2022

@author: baly
"""

import torch
import sys
import os 
import numpy as np
import ot
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
device='cpu'

# load data
from sopt2.library import *
from sopt2.sliced_opt import *
root='experiment/dataset_similarity/' 
num='1'
data=torch.load(root+'data/data'+num+'.pt')
set_A=data['A']
set_B=data['B']
set_C=data['C']

# load encoder 
size=200
#N=n*200
print('OT method')
n=3

result={}
#set_C[0:400]=set_A[0:400]
#n=3
set_A=set_A.reshape((600,28*28))
set_B=set_B.reshape((600,28*28))
set_C=set_C.reshape((600,28*28))
print('OT method')
mu=np.ones(size*n)
nu=np.ones(size*n)

cost_M=cost_matrix_T(set_A,set_B)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.lp.emd(mu,nu,cost_M1)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,B',loss)
result['OT']['AB']=loss
cost_M=cost_matrix_T(set_A,set_C)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.lp.emd(mu,nu,cost_M1)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,C',loss)
result['OT']['AC']=loss


print('OPT method')
Mass=(n-1)*200
cost_M=cost_matrix_T(set_A,set_B)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, Mass, nb_dummies=1, log=False)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,B',loss)

result['OPT']['AB']=loss


cost_M=cost_matrix_T(set_A,set_C)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, Mass, nb_dummies=1, log=False)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,C',loss)
result['OPT']['AC']=loss



print('sopt method')
n_projections=20*50
Lambda=5e-5
sopt_M=sopt(set_A,set_B,Lambda,n_projections,'orth')
sopt_M.get_projections()
sopt_M.get_plans()
loss,mass=sopt_M.sliced_cost(penulty=True)
print('trasported mass',mass)
print('distance between A,B',loss*n_projections)
result['Lambda']=Lambda
result['SOPT']['AB']=loss
redult['SOPT']['AB mass']=mass


sopt_M=sopt(set_A,set_C,Lambda,n_projections,'orth')
sopt_M.get_projections()
sopt_M.get_plans()
loss,mass=sopt_M.sliced_cost(penulty=True)
print('transported mass',mass)
print('distance between A,C',loss*n_projections)


result['SOPT']['AC']=loss
redult['SOPT']['AC mass']=mass
