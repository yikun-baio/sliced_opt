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
L1=[[0,1,2],[0,4,7],[0,1,0]]
L2=[[1,2,5],[1,8,9],[1,2,8]]
L3=[[3,5,8],[3,0,2],[3,5,3]]
L4=[[1,7,9],[1,0,4],[1,9,6]]
L5=[]

# load data
from sopt2.library import *
from sopt2.sliced_opt import *
root='experiment/dataset_similarity/' 
num='5'
data=torch.load(root+'data/test/data'+num+'.pt')
set_A=data['Ae'].detach()#[0:400]
set_B=data['Be'].detach()#[0:400]
set_C=data['Ce'].detach()#[0:400]



# load encoder 
size=200
#N=n*200

n=3

result={}
#set_C[0:400]=set_A[0:400]
#n=3
#set_A=set_A.reshape((200,28*28))
#set_B=set_B.reshape((200,28*28))
#set_C=set_C.reshape((200,28*28))


print('OT method')
mu=np.ones(size*n-00)
nu=np.ones(size*n-00)

cost_M=cost_matrix_T(set_A,set_B)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.lp.emd(mu,nu,cost_M1)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,B',loss)
result['OT-AB']=loss
cost_M=cost_matrix_T(set_A,set_C)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.lp.emd(mu,nu,cost_M1)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,C',loss)
result['OT-AC']=loss


print('OPT method')
Mass=200
cost_M=cost_matrix_T(set_A,set_B)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, Mass, nb_dummies=1, log=False)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,B',loss)

result['OPT-AB']=loss


cost_M=cost_matrix_T(set_A,set_C)
cost_M1=cost_M.detach().cpu().numpy()
plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, Mass, nb_dummies=1, log=False)
plan_T=torch.from_numpy(plan).to(device)
loss=torch.sum(cost_M*plan_T)
print('distance between A,C',loss)
result['OPT-AC']=loss



print('sopt method')
n_projections=32*50
Lambda=3e-7
sopt_M=sopt(set_A,set_B,Lambda,n_projections,'orth')
sopt_M.get_projections()
sopt_M.get_plans()
loss,mass=sopt_M.sliced_cost(penulty=True)
print('trasported mass',mass)
print('distance between A,B',loss*n_projections)
result['Lambda']=Lambda
result['SOPT-n_projections']=n_projections
result['SOPT-AB']=loss
result['SOPT-AB mass']=mass


sopt_M=sopt(set_A,set_C,Lambda,n_projections,'orth')
sopt_M.get_projections()
sopt_M.get_plans()
loss,mass=sopt_M.sliced_cost(penulty=True)
print('transported mass',mass)
print('distance between A,C',loss*n_projections)


result['SOPT-AC']=loss
result['SOPT-AC mass']=mass

print('SOT method')


mu=np.ones(size*n)
nu=np.ones(size*n)


loss=ot.sliced.sliced_wasserstein_distance(set_A,set_B,n_projections=n_projections,p=2)
print('distance between A,B',loss)
result['SOT-AB']=loss


loss=ot.sliced.sliced_wasserstein_distance(set_A,set_C,n_projections=n_projections,p=2)
print('distance between A,C',loss)
result['SOT-AC']=loss

torch.save(result,root+'result/'+num+'.pt')
