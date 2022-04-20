# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:27:59 2022

@author: laoba
"""



import torch
import numpy as np 
from OPT import *
from library import *
import ot 
import matplotlib.pyplot as plt
import time

def OPT_refined(X,Y,L):
    L_x=[i for i,j in enumerate(L) if j>=0]
    L_y=[j for j in L if j>=0]
    num_destruction=len(L)-len(L_x)
    cost=sum(cost_function(X[L_x],Y[L_y]))+2*Lambda*num_destruction
    return cost
# Experiment 1 Relation between OT and OPT

Lambda=8


time1_list=[]
time2_list=[]
time3_list=[]
time4_list=[]

start_n=10
end_n=35

for n in range (start_n,end_n,5):
    m=n+10
    X=np.random.uniform(0,20,size=n)
    Y=np.random.uniform(5,30,size=m)
    mu=np.ones(n)
    nu=np.ones(m)
    
    cost_matrix=np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            cost_matrix[i,j]=cost_function(X[i],Y[j])

    
    X.sort()
    Y.sort()
    
    start_time = time.time()
    cost1,L1=OPT_1D_v1(X,Y,Lambda)
    end_time = time.time()
    time1=end_time-start_time 
    
    start_time = time.time()
    cost2,L2=OPT_1D_v3(X,Y,Lambda)
    end_time = time.time()
    time2=end_time-start_time
    
    start_time = time.time()
    cost3,L3=POT_1D(X,Y)
    end_time = time.time()
    time3=end_time-start_time
    
    L_y=[j for j in L2 if j>=0]
    start_time = time.time()
    cost4=ot.partial.entropic_partial_wasserstein(mu,nu,cost_matrix,len(L_y))
    end_time = time.time()
    time4=end_time-start_time
    
    
    time1_list.append(time1)
    time2_list.append(time2)
    time3_list.append(time3)
    time4_list.append(time4)

    
    


plt.semilogy(range(start_n,end_n),time1_list,label='OPT v1')
plt.semilogy(range(start_n,end_n),time2_list,label='OPT v2')
plt.semilogy(range(start_n,end_n),time3_list,label='POT')
plt.semilogy(range(start_n,end_n),time4_list,label='Sinkhon in POT package')

plt.xlabel("n: size of X")
plt.ylabel("runing time")
plt.legend(loc='best')
plt.show()

# plt.plot(range(start_n,end_n,5),time2_list,label='OPT v2')
# plt.plot(range(start_n,end_n,5),time3_list,label='POT')
# plt.plot(range(start_n,end_n,5),time4_list,label='Sinkhorn in POT package')
# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()