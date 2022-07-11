# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:27:59 2022

@author: laoba
"""


import torch
import numpy as np 
import os
import sys




work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
device='cpu'

# load data


from sopt1.opt import *
from sopt1.library import *
import ot 
import matplotlib.pyplot as plt
import time


Lambda=10

time1_list=[]
time2_list=[]
time3_list=[]
time4_list=[]
time5_list=[]

start_n=100
end_n=1000
device='cpu'
step=100
k=5
for n in range (start_n,end_n,step):
    m=n+10

    time1=0
    time2=0
    time3=0
    time4=0
    time5=0
    mu=torch.ones(n,device='cpu')
    nu=torch.ones(m,device='cpu')
    for i in range (k):
        torch.manual_seed(0)
        X=0+(20-0)*torch.rand(n,device=device)
        Y=torch.rand(m,device=device)*(30-5)+5
        X1=X.numpy().copy()
        Y1=Y.numpy().copy()
        
        start_time = time.time()
        X1.sort()
        Y1.sort()        
        cost1,L1=pot_1d(X1,Y1)
        end_time = time.time()
        time1+=end_time-start_time
        
        X1=X.numpy().copy()
        Y1=Y.numpy().copy()
        start_time = time.time()
        X1.sort()
        Y1.sort()       
        cost2,L2=opt_1d_v2(X1,Y1,Lambda)
        end_time = time.time()
        time2+=end_time-start_time
    
    
        L_y=[j for j in L2 if j>=0]
        X1=X.numpy().copy()
        Y1=Y.numpy().copy()
        start_time = time.time()
        M=cost_matrix(X1,Y1)
        L3=ot.partial.entropic_partial_wasserstein(mu,nu,M,len(L_y))
        end_time = time.time()
        time3+=end_time-start_time
    
    

    # start_time = time.time()
    # M=cost_matrix(X,Y)
    # L5=ot.partial.partial_wasserstein(mu,nu,M,len(L_y))
    # end_time = time.time()
    # time5=end_time-start_time
    
    
    time1_list.append(time1/k)
    time2_list.append(time2/k)
    time3_list.append(time3/k)
#    time4_list.append(time4/k)
    # time5_list.append(time5)

    
    


plt.semilogy(range(start_n,end_n,step),time1_list,label='partial OT')
plt.semilogy(range(start_n,end_n,step),time2_list,label='ours v2')
plt.semilogy(range(start_n,end_n,step),time3_list,label='Sinkhorn')
plt.legend(loc='best')
plt.xlabel('n: size of X')
plt.ylabel("runing time")
plt.show()
#plt.semilogy(range(start_n,end_n),time4_list,label='Sinkhon in POT package')

# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()
plt.plot(range(start_n,end_n,step),time1_list,label='partial ot')
plt.plot(range(start_n,end_n,step),time2_list,label='ours v2 ')
plt.plot(range(start_n,end_n,step),time3_list,label='Sinkhorn: python ot')
#plt.plot(range(start_n,end_n,step),time5_list,label='LP: python ot')
plt.xlabel("n: size of X")
plt.ylabel("runing time")
plt.legend(loc='best')
plt.show()