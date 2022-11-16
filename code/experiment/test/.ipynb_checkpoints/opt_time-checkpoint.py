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


from sopt.opt import *
from sopt.library import *
from sopt.lib_ot import *
import ot 
import matplotlib.pyplot as plt
import time




#Lambda=60
Lambda_list=np.array([20.0,100.0]).astype(np.float64)
time_pot_list=[]
time_v2_list=[[],[]]
time_v2_a_list=[[],[]]
time_lp_list=[[],[]]
time_new_list=[[],[]]
#time5_list=[]

start_n=1500
end_n=10000
device='cpu'
step=500
k=10
print('start')



for n in range (start_n,end_n,step):

    m=n+1000
    print('n',n)
    print('m',m)
    time_pot=0
    time_v2=np.zeros(2)
    time_v2_a=np.zeros(2)
    time_lp=np.zeros(2)
    time_new=np.zeros(2)
    mu=np.ones(n)
    nu=np.ones(n)

    for i in range (k):
        X=np.random.uniform(-20,20,n).astype(np.float64)
        Y=np.random.uniform(-40,40,m).astype(np.float64)
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

#             X1.sort()
#             Y1.sort()       
#             opt_1d_v2(X1,Y1,Lambda)
#             end_time = time.time()
#             time_v2[j]+=end_time-start_time
            
#             X1=X.copy()
#             Y1=Y.copy()
#             start_time = time.time()
# #            M=cost_matrix(X1,Y1)
#             X1.sort()
#             Y1.sort()
#             opt_1d_v2_a(X1,Y1,Lambda)
# #        L3=ot.partial.entropic_partial_wasserstein(mu,nu,M,len(L_y))
#             end_time = time.time()
#             time_v2_a[j]+=end_time-start_time
            
            
            start_time = time.time()
            X1.sort()
            Y1.sort()       
#            M=cost_matrix(X1,Y1)
            phi,psi,piRow,piCol=solve_opt(X1,Y1,Lambda) #,verbose=False,plots=False)
            L_new=getPiFromRow(n,m,piRow)
            end_time = time.time()
            time_new[j]+=end_time-start_time
    
            
    #        Y1=Y[0:n]
            numItermax=200000*n/1000
            start_time = time.time()
            opt_lp(X,Y,Lambda,numItermax)
            end_time = time.time()
            time_lp[j]+=end_time-start_time
    
    
    time_pot_list.append(time_pot/k)
    for j in range(2):
        time_v2_list[j].append(time_v2[j]/k)
        time_v2_a_list[j].append(time_v2_a[j]/k)
        time_lp_list[j].append(time_lp[j]/k)
        time_new_list[j].append(time_new[j]/k)

time_list={}
time_list['pot']=time_pot_list
time_list['v2']=time_v2_list
time_list['v2_a']=time_v2_a_list
time_list['lp']=time_lp_list 
time_list['new']=time_new_list

torch.save(time_list,'experiment/test/results/time_list_numba.pt')

time_list=torch.load('experiment/test/results/time_list_numba.pt')
time_pot_list=time_list['pot']
time_v2_list=time_list['v2']
time_v2_a_list=time_list['v2_a']
time_lp_list=time_list['lp']
time_new_list=time_list['new']

start_n=1500
end_n=10000
device='cpu'
step=500
k=10
n_list=range(start_n,end_n,step)[1:]
fig = plt.figure()
ax = plt.subplot(111)

plt.semilogy(n_list,time_pot_list[1:],label='partial OT')
for j in range(2):
#    plt.semilogy(n_list,time_v2_list[j][1:],label='ours,$\lambda=$'+str(Lambda_list[j]))
#    plt.semilogy(n_list,time_v2_a_list[j][1:],label='ours_a,$\lambda=$'+str(Lambda_list[j]))
    plt.semilogy(n_list,time_lp_list[j][1:],label='lp: python ot, C, $\lambda=$'+str(Lambda_list[j]))
    plt.semilogy(n_list,time_new_list[j][1:],label='ours, $\lambda=$'+str(Lambda_list[j]))
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.23),
          fancybox=True, shadow=True, ncol=3)
plt.xlabel('n: size of X')
plt.ylabel("wall time")
plt.savefig('experiment/test/results/time_numba.png',format='png',dpi=800,bbox_inches='tight')
#plt.title('wall-clock time with accelaration')
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