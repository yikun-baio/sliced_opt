# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:03:27 2022

@author: laoba
"""


import torch
import numpy as np 
from OPT import *
from library import *
import ot 
import matplotlib.pyplot as plt


def OPT_refined(X,Y,L):
    L_x=[i for i,j in enumerate(L) if j>=0]
    L_y=[j for j in L if j>=0]
    num_destruction=len(L)-len(L_x)
    cost=sum(cost_function(X[L_x],Y[L_y]))+2*Lambda*num_destruction
    return cost
# Experiment 1 Relation between OT and OPT
n=20
m=30
Lambda=8
num_expriment=100
cost1_list=[]
cost2_list=[]
error_list=[]


for i in range (0,num_expriment):
    X=np.random.uniform(0,20,size=n)
    Y=np.random.uniform(5,30,size=m)
    X.sort()
    Y.sort()
    cost1,L1=OPT_1D_v1(X,Y,Lambda)
    cost2,L2=OPT_1D_v3(X,Y,Lambda)
    cost3=OPT_refined(X,Y,L1)
    cost1_list.append(cost1)
    cost2_list.append(cost2)
    if L1!=L2:
        print('error')

plt.plot(range(0,num_expriment),cost1_list,label='OPT version 1')
plt.plot(range(0,num_expriment),cost2_list,label='OPT version 2')
plt.xlabel("# of experiment")
plt.ylabel("Distance")
plt.title("OPT v1 vs v2")
plt.legend()
plt.show()

scalar_list=np.linspace(1,100,100)
for scalar in scalar_list:
    X=np.random.uniform(0,10,size=n)
    Y=np.random.uniform(5,20,size=m)
    X=X*scalar
    Y=Y*scalar
    Lambda=8*scalar
    X.sort()
    Y.sort()
    cost1,L1=OPT_1D_v1(X,Y,Lambda)
    cost2,L2=OPT_1D_v3(X,Y,Lambda)
    error_list.append(abs(cost2-cost1))

    

plt.semilogy(scalar_list,error_list)
plt.xlabel("scalar")
plt.ylabel("|distance 1-distance 2|")
plt.title("OPT v1 vs v2")
plt.legend()
plt.show()


 