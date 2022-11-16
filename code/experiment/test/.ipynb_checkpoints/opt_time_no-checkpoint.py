#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:22:02 2022

@author: baly
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

from sopt.lib_ot import *
#Lambda=60
Lambda_list=np.float32(np.array([20.0,100.0]))
time_pot_list=[]
time_v2_list=[[],[]]
time_v2_a_list=[[],[]]
time_sinkhorn_list=[[],[]]
time_new_list=[[],[]]

start_n=200
end_n=1200
device='cpu'
step=100
k=5
for n in range (start_n,end_n,step):
    print('n',n)
    m=n+1000
    time_pot=0
    time_v2=np.zeros(2)
    time_v2_a=np.zeros(2)
    time_sinkhorn=np.zeros(2)
    time_new=np.zeros(2)
    mu=np.ones(n)
    nu=np.ones(m)
    for i in range (k):
        X=np.float32(np.random.uniform(-20,20,n))
        Y=np.float32(np.random.uniform(-40,40,m))
        X1=X.copy()
        Y1=Y.copy()
        start_time = time.time()
        X1.sort()
        Y1.sort()        
        cost1,L1=pot_1d(X1,Y1)
        end_time = time.time()
        time_pot+=end_time-start_time
        
        for j in range(len(Lambda_list)):
            Lambda=Lambda_list[j]
            X1=X.copy()
            Y1=Y.copy()
            start_time = time.time()
            X1.sort()
            Y1.sort()       
            cost,L=opt_1d_v2(X1,Y1,Lambda)
            end_time = time.time()
            time_v2[j]+=end_time-start_time
            mass=np.sum(L>=0)

            #=np.sum(L2>=0)
            X1=X.copy()
            Y1=Y.copy()
            start_time = time.time()
            M=cost_matrix(X1,Y1)
            X1.sort()
            Y1.sort()       
            opt_1d_v2_apro(X1,Y1,Lambda)
            end_time = time.time()
            time_v2_a[j]+=end_time-start_time
            
    
    #        Y1=Y[0:n]
            start_time = time.time()
            M=cost_matrix(X1,Y1)
            ot.partial.entropic_partial_wasserstein(mu,nu,M,mass)
            end_time = time.time()
            time_sinkhorn[j]+=end_time-start_time
            
            
            start_time = time.time()
            X1.sort()
            Y1.sort()       
#            M=cost_matrix(X1,Y1)
            phi,psi,piRow,piCol=solve1DOPT_32_no(X1,Y1,Lambda/2) #,verbose=False,plots=False)
            L_new=getPiFromRow(n,m,piRow)
            end_time = time.time()
            time_new[j]+=end_time-start_time
            
    
    
    time_pot_list.append(time_pot/k)
    for j in range(2):
        time_v2_list[j].append(time_v2[j]/k)
        time_v2_a_list[j].append(time_v2_a[j]/k)
        time_sinkhorn_list[j].append(time_sinkhorn[j]/k)
        time_new_list[j].append(time_new[j]/k)

time_list={}
time_list['pot']=time_pot_list
time_list['v2']=time_v2_list
time_list['v2_a']=time_v2_a_list
time_list['sinkhorn']=time_sinkhorn_list 
time_list['new']=time_sinkhorn_list 

torch.save(time_list,'experiment/test/results/time_list_no.pt')


start_n=200
end_n=1200
device='cpu'
step=100
k=5
time_list=torch.load('experiment/test/results/time_list_no.pt')
time_pot_list=time_list['pot']
time_v2_list=time_list['v2']
time_v2_a_list=time_list['v2_a']
time_sinkhorn_list=time_list['sinkhorn']
time_new=time_list['new']

n_list=range(start_n,end_n,step)
fig = plt.figure()
ax = plt.subplot(111)

plt.semilogy(n_list,time_pot_list,label='partial OT')
for j in range(2):
#    plt.semilogy(n_list,time_v2_list[j],label='ours-$\lambda=$'+str(int(Lambda_list[j])))
#    plt.semilogy(n_list,time_v2_a_list[j],label='ours_a-$\lambda=$'+str(int(Lambda_list[j])))
    plt.semilogy(n_list[:],time_sinkhorn_list[j],label='sinkhorn-$\lambda=$'+str(int(Lambda_list[j])))
    plt.semilogy(n_list,time_new_list[j],label='ours-$\lambda=$'+str(int(Lambda_list[j])))
    
    
#plt.semilogy(n_list,time_sinkhorn_list,label='Sinkhorn: python ot')
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.22),
          fancybox=True, shadow=True, ncol=3)
plt.xlabel('n: size of X')
plt.ylabel("wall time")
#plt.title('wall-clock time without accelaration')
plt.savefig('experiment/test/results/time_no.png',format='png',dpi=800,bbox_inches='tight')
plt.show()
#plt.semilogy(range(start_n,end_n),time4_list,label='Sinkhon in POT package')

# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()
# plt.plot(range(start_n,end_n,step),time1_list,label='partial ot')
# plt.plot(range(start_n,end_n,step),time2_list,label='ours v2 ')
# plt.plot(range(start_n,end_n,step),time3_list,label='ours v2-apro')
# plt.plot(range(start_n,end_n,step),time5_list,label='LP: python ot, C')
# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()