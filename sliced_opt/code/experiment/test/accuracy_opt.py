# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:11:13 2022

@author: laoba
"""


import torch
import numpy as np 
import os
import sys
import ot
import matplotlib.pyplot as plt
import time

work_path=os.path.dirname(__file__)
print('work_path is', work_path)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)


lab_path=parent_path
os.chdir(lab_path)
sys.path.append(parent_path)

from sopt2.opt import *
from sopt2.library import *



Lambda=8.0

start_n=20
end_n=200
step=5
cost1_list=[]
cost2_list=[]
cost3_list=[]
cost4_list=[]
for n in range (start_n,end_n,step):
    m=n+10

    X1=2*torch.rand(n,dtype=torch.float32)
    Y1=3*torch.rand(m,dtype=torch.float32)-0.3
    X1=X1.sort().values
    Y1=Y1.sort().values
    mu=np.ones(n)
    nu=np.ones(m)
    X=X1.numpy()
    Y=Y1.numpy()    
    cost1,L1=opt_1d_np(X,Y,Lambda)
    cost2,L2=opt_1d_T(X1,Y1,Lambda)


    mass=np.sum(L1>=0) 
    M=cost_matrix(X,Y)
#    L3=ot.partial.partial_wasserstein
    L3=ot.partial.partial_wasserstein(mu,nu,M,mass)
    cost3=sum(sum(M*L3))
    cost3+=(n-mass)*Lambda
    

    M=cost_matrix(X,Y)
    L4=ot.partial.entropic_partial_wasserstein(mu,nu,M,0.1,mass)
    cost4=sum(sum(M*L4))+(n-mass)*Lambda
    
    cost1_list.append(cost1)
    cost2_list.append(cost2)
    cost3_list.append(cost3)
    cost4_list.append(cost4)
     
    


plt.plot(range(start_n,end_n,step),cost1_list,'-',label='ours v2 np')
plt.plot(range(start_n,end_n,step),cost2_list,label='ours v2 torch')
plt.plot(range(start_n,end_n,step),cost3_list,label='Lp: python OT')
#plt.plot(range(start_n,end_n,step),cost4_list,label='Sinkhorn: python OT')
plt.xlabel("n: size of X")
plt.ylabel("OPT distances")
plt.legend(loc='best')
plt.show()

#plt.plot(range(start_n,end_n),cost4_list,label='Sinkhorn: python OT')
# plt.semilogy(range(start_n,end_n),time3_list,label='POT')
# plt.semilogy(range(start_n,end_n),time4_list,label='Sinkhon in POT package')



# plt.plot(range(start_n,end_n,10),time2_list,label='out OPT')
# plt.plot(range(start_n,end_n,10),time3_list,label='Partial OT')
# plt.plot(range(start_n,end_n,10),time4_list,label='Sinkhorn')
# plt.xlabel("n: size of X")
# plt.ylabel("runing time")
# plt.legend(loc='best')
# plt.show()