# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:11:13 2022

@author: laoba
"""


import torch
import numpy as np 
import os

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
lab_path=parent_path+'\\sopt'
os.chdir(lab_path)

from opt import *
from library import *
import ot 
import matplotlib.pyplot as plt
import time

def OPT_refined(X,Y,L):
    L_x=[i for i,j in enumerate(L) if j>=0]
    L_y=[j for j in L if j>=0]
    num_destruction=len(L)-len(L_x)
    cost=sum(cost_function(X[L_x],Y[L_y]))
    return cost

def OPT_refined_T(X,Y,L):
    L_x=[i for i,j in enumerate(L) if j>=0]
    L_y=[j for j in L if j>=0]
    num_destruction=len(L)-len(L_x)
    cost=sum(cost_function(X[L_x],Y[L_y]))
    return cost
# Experiment 1 Relation between OT and OPT

Lambda=8.0

start_n=20
end_n=30

cost1_list=[]
cost2_list=[]
cost3_list=[]
cost4_list=[]
for n in range (start_n,end_n):
    m=n+10
    X=np.random.uniform(0,20,size=n)
    Y=np.random.uniform(5,30,size=m)
#    X=(20-0)*torch.rand(n,device='cuda')+0
#    Y=(30-5)*torch.rand(m,device='cuda')+5
    mu=np.ones(n)
    nu=np.ones(m)
    X.sort()
    Y.sort()
        
    cost1,L1=opt_1d_np(X,Y,Lambda)
    cost1_trans=OPT_refined(X,Y,L1)

    cost2,L2=opt_1d_v2(X,Y,Lambda)
    cost2_trans=OPT_refined(X,Y,L2)

    L_y=[j for j in L2 if j>=0]
    mass=float(len(L_y))
    M=cost_matrix(X,Y)
    L3=ot.partial.partial_wasserstein(mu,nu,M,mass)
    cost3=sum(sum(M*L3))
    
    L4=ot.partial.entropic_partial_wasserstein(mu,nu,M,0.1,mass)
    cost4=sum(sum(M*L4))
    
    cost1_list.append(cost1_trans)
    cost2_list.append(cost2_trans)
    cost3_list.append(cost3)
    cost4_list.append(cost4)
     
    



plt.plot(range(start_n,end_n),cost3_list,label='Lp: python OT')
plt.plot(range(start_n,end_n),cost1_list,'-',label='ours v2 np')
plt.plot(range(start_n,end_n),cost2_list,label='ours v2')

#plt.plot(range(start_n,end_n),cost4_list,label='Sinkhorn: python OT')
# plt.semilogy(range(start_n,end_n),time3_list,label='POT')
# plt.semilogy(range(start_n,end_n),time4_list,label='Sinkhon in POT package')

plt.xlabel("n: size of X")
plt.ylabel("OPT distances")
plt.legend(loc='best')
plt.show()

# plt.plot(range(start_n,end_n,10),time2_list,label='out OPT')
# plt.plot(range(start_n,end_n,10),time3_list,label='Partial OT')
# plt.plot(range(start_n,end_n,10),time4_list,label='Sinkhorn')
# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()