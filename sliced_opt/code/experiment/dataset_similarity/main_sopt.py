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

from sopt2.library import *
from sopt2.sliced_opt import *
root='experiment/dataset_similarity/data'
set_A=torch.load(root+'/set_A.pt')
set_B=torch.load(root+'/set_B.pt')
set_C=torch.load(root+'/set_C.pt')
#set_C[0:200]=set_A[0:200]

print('OT method')
mu=np.ones(300)
nu=np.ones(300)
cost_M=cost_matrix_T(set_A,set_B)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.lp.emd(mu,nu,cost_M1)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print(loss)


cost_M=cost_matrix_T(set_A,set_C)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.lp.emd(mu,nu,cost_M1)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print(loss)

print('OPT method')
cost_M=cost_matrix_T(set_A,set_B)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, m=200, nb_dummies=1, log=False)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print(loss)

cost_M=cost_matrix_T(set_A,set_C)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, m=200, nb_dummies=1, log=False)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print(loss)



print('sopt method')
n_projections=1800
Lambda=3e-4
A=sopt_for(set_A,set_B,Lambda,n_projections,'orth')


loss,mass=A.sliced_cost(penulty=True)
print(mass)
print(loss)


#Lambda=2.15e-5
A=sopt_for(set_A,set_C,Lambda,n_projections,'orth')
loss,mass=A.sliced_cost(penulty=True)
print(mass)
print(loss)

