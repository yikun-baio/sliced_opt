#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:24:17 2022

@author: baly
"""


import torch
import sys
import os 
import numpy as np
import ot
import pandas as pd
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
num='test'
Lambda='_1'
data=torch.load(root+'data/data'+num+Lambda+'.pt')
set_A=data['A']
set_B=data['B']


# load encoder 
size=200
#N=n*200
print('OT method')
n=20
set_A=set_A.reshape((size*n,28*28))
set_B=set_B.reshape((size*n,28*28))
# set_C=set_C.reshape((size*n,28*28))

cost_list=torch.zeros(20,20)
for i in range(20):
    for j in range(20):
        A=set_A[size*i:size*(i+1)].reshape(size,28*28)
        B=set_B[size*j:size*(j+1)].reshape(size,28*28)
        mu=np.ones(size)/size
        nu=np.ones(size)/size
        cost_M=cost_matrix_T(A,B)
        cost_M1=cost_M.detach().cpu().numpy()
        plan=ot.lp.emd(mu,nu,cost_M1)
        plan_T=torch.from_numpy(plan).to(device)
        loss=torch.sum(cost_M*plan_T)
        cost_list[i,j]=loss
        if i==j:
            print('i=',i)
            print('distance between A,B',loss)


set_Ae=data['Ae']
set_Be=data['Be']
coste_list=torch.zeros(20,20)
for i in range(20):
    for j in range(20):
        Ae=set_Ae[size*i:size*(i+1)]
        Be=set_Be[size*j:size*(j+1)]
        mu=np.ones(size)/size
        nu=np.ones(size)/size
        cost_M=cost_matrix_T(Ae,Be)
        cost_M1=cost_M.detach().cpu().numpy()
        plan=ot.lp.emd(mu,nu,cost_M1)
        plan_T=torch.from_numpy(plan).to(device)
        loss=torch.sum(cost_M*plan_T)
        coste_list[i,j]=loss
        if i==j:
            print('distance between Ae,Be',loss)

List={}
List['cost_list']=cost_list
List['coste_list']=coste_list
torch.save(List,root+'data/costlist_1.pt')



labels=["0","1", "2", "3","4","5","6","7","8","9","O",'I',"II","III","IV","V","VI","VII","VIII","IX"]
df = pd.DataFrame(cost_list).T
df.to_excel(excel_writer = root+"data/test.xlsx")
df = pd.DataFrame(coste_list.detach()).T
df.to_excel(excel_writer = root+"data/teste.xlsx")

# #set_C[0:400]=set_A[0:400]
# #n=3
# print('OT method')
# mu=np.ones(size*n)
# nu=np.ones(size*n)

# cost_M=cost_matrix_T(set_A,set_B)
# cost_M1=cost_M.detach().cpu().numpy()
# plan=ot.lp.emd(mu,nu,cost_M1)
# plan_T=torch.from_numpy(plan).to(device)
# loss=torch.sum(cost_M*plan_T)
# print('distance between A,B',loss)

# cost_M=cost_matrix_T(set_A,set_C)
# cost_M1=cost_M.detach().cpu().numpy()
# plan=ot.lp.emd(mu,nu,cost_M1)
# plan_T=torch.from_numpy(plan).to(device)
# loss=torch.sum(cost_M*plan_T)
# print('distance between A,C',loss)



# print('OPT method')
# Mass=(n-1)*200
# cost_M=cost_matrix_T(set_A,set_B)
# cost_M1=cost_M.detach().cpu().numpy()
# plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, Mass, nb_dummies=1, log=False)
# plan_T=torch.from_numpy(plan).to(device)
# loss=torch.sum(cost_M*plan_T)
# print('distance between A,B',loss)

# cost_M=cost_matrix_T(set_A,set_C)
# cost_M1=cost_M.detach().cpu().numpy()
# plan=ot.partial.partial_wasserstein(mu, nu, cost_M1, Mass, nb_dummies=1, log=False)
# plan_T=torch.from_numpy(plan).to(device)
# loss=torch.sum(cost_M*plan_T)
# print('distance between A,C',loss)



# print('sopt method')
# n_projections=32*100
# Lambda=9e-6
# sopt_M=sopt(set_A,set_B,Lambda,n_projections,'orth')
# sopt_M.get_projections()
# sopt_M.get_plans()
# loss,mass=sopt_M.sliced_cost(penulty=True)
# print('trasported mass',mass)
# print('distance between A,B',loss*n_projections)

# Lambda=9target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10

# sopt_M=sopt(set_A,set_C,Lambda,n_projections,'orth')
# sopt_M.get_projections()
# sopt_M.get_plans()
# loss,mass=sopt_M.sliced_cost(penulty=True)
# print('transported mass',mass)
# print('distance between A,C',loss*n_projections)

