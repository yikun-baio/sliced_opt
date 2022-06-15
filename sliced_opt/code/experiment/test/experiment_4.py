# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:27:59 2022

@author: laoba
"""

#!%%pypy
#%%pypy
import torch
import numpy as np 
import os

parent_path='/home/yikun/Dropbox/Yikun-Bai/Sliced-Optimal-Partial-Transport/code3/'
#parent_path='C:/Users/laoba/Dropbox/Yikun-Bai/Sliced-Optimal-Partial-Transport/Code/'
os.chdir(parent_path)
#current_path = os.path.dirname(os.path.realpath(__file__))
#parent_path = os.path.dirname(current_path)
#lab_path=parent_path+'\\sopt'
#os.chdir(lab_path)

from opt import *
from library import *
import ot 
import matplotlib.pyplot as plt
import time


Lambda=10

time1_list=[]
time2_list=[]
time3_list=[]
time4_list=[]
time5_list=[]

start_n=1000
end_n=2000
device='cpu'
step=100
k=1
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
        X=0+(20-0)*torch.rand(n,device=device).requires_grad_()
        Y=torch.rand(m,device=device)*(30-5)+5
        
        start_time = time.time()
        X=X.sort().values
        Y=Y.sort().values
        X1=list_to_numpy_array(X)
        Y1=list_to_numpy_array(Y)
        cost1,L1=pot_1d(X1,Y1)
        end_time = time.time()
        time1+=end_time-start_time
        
        start_time = time.time()
        X=X.sort().values
        Y=Y.sort().values
        X1=list_to_numpy_array(X)
        Y1=list_to_numpy_array(Y)
        cost2,L2=opt_1d_np(X1,Y1,Lambda)
        end_time = time.time()
        time2+=end_time-start_time
    
        # start_time = time.time()
        # X=X.sort().values
        # Y=Y.sort().values
        # cost3,L3=opt_1d_T(X,Y,Lambda)
        # end_time = time.time()
        # time3+=end_time-start_time
    
    
        L_y=[j for j in L2 if j>=0]
        start_time = time.time()
        # X=list_to_numpy_array(X)
        # Y=list_to_numpy_array(Y)
  #      print(X)
    #    print(Y)
 #       
        # M=cost_matrix_T(X,Y).to(device='cpu')
        # L4=ot.partial.entropic_partial_wasserstein(mu,nu,M,len(L_y))
        # end_time = time.time()
        # time4+=end_time-start_time
    
    

    # start_time = time.time()
    # M=cost_matrix(X,Y)
    # L5=ot.partial.partial_wasserstein(mu,nu,M,len(L_y))
    # end_time = time.time()
    # time5=end_time-start_time
    
    
    time1_list.append(time1/k)
    time2_list.append(time2/k)
    time3_list.append(time3/k)
    # time4_list.append(time4/k)
    # time5_list.append(time5)

    
    


# plt.semilogy(range(start_n,end_n),time1_list,label='OPT v1')
# plt.semilogy(range(start_n,end_n),time2_list,label='OPT v2')
# plt.semilogy(range(start_n,end_n),time3_list,label='POT')
# plt.semilogy(range(start_n,end_n),time4_list,label='Sinkhon in POT package')

# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()
plt.plot(range(start_n,end_n,step),time1_list,label='partial ot')
plt.plot(range(start_n,end_n,step),time2_list,label='ours v2 np')
#plt.plot(range(start_n,end_n,step),time3_list,label='ours v2 torch '+device)
# plt.plot(range(start_n,end_n,step),time4_list,label='Sinkhorn: python ot')
#plt.plot(range(start_n,end_n,step),time5_list,label='LP: python ot')
plt.xlabel("n: size of X")
plt.ylabel("runing time")
plt.legend(loc='best')
plt.show()