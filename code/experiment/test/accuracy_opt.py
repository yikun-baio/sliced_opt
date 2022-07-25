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



Lambda=np.float32(10.0)

start_n=100
end_n=1000
step=5
cost1_list=[]
cost2_list=[]
cost3_list=[]
cost4_list=[]
for n in range (start_n,end_n,step):
    m=n+200

#    X1=2*torch.rand(n,dtype=torch.float32)
#    Y1=3*torch.rand(m,dtype=torch.float32)-0.3
#    X1=X1.sort().values
#    Y1=Y1.sort().values
    X=np.float32(np.random.uniform(-20,20,n))
    Y=np.float32(np.random.uniform(-40,40,m))
    X.sort()
    Y.sort()
    mu=np.ones(n)
    nu=np.ones(m)
    
    cost1,L1=opt_1d_v2_apro(X,Y,Lambda)
    cost2,L2=opt_1d_v2(X,Y,Lambda)


    mass=np.sum(L2>=0) 
    M=cost_matrix(X,Y)
#    L3=ot.partial.partial_wasserstein
    L3=ot.partial.partial_wasserstein(mu,nu,M,mass)
    cost3=sum(sum(M*L3))
    cost3+=(n-mass)*Lambda
    

    M=cost_matrix(X,Y)
#    L4=ot.partial.entropic_partial_wasserstein(mu,nu,M,0.1,mass)
#    cost4=sum(sum(M*L4))+(n-mass)*Lambda
    
    cost1_list.append(cost1)
    cost2_list.append(cost2)
    cost3_list.append(cost3)
#    cost4_list.append(cost4)
     
    


plt.plot(range(start_n,end_n,step),cost1_list,'-',label='ours v2-apro')
plt.plot(range(start_n,end_n,step),cost2_list,label='ours v2')
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