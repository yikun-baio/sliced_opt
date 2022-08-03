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
from sopt2.lib_ot import *



Lambda=np.float32(10.0)

start_n=100
end_n=1000
step=5
cost_v2_list=[]
cost_v2_a_list=[]
cost_pr_list=[]
cost_lp_list=[]
cost_sinkhorn_list=[]
for n in range (start_n,end_n,step):
    m=n+200

    X=np.float32(np.random.uniform(-20,20,n))
    Y=np.float32(np.random.uniform(-40,40,m))
    X.sort()
    Y.sort()
    mu=np.ones(n)
    nu=np.ones(m)
    
    cost_v2_a,L_v2_a=opt_1d_v2_apro(X,Y,Lambda)
    cost_v2,L_v2=opt_1d_v2(X,Y,Lambda)


    mass=np.sum(L_v2>=0) 
    M=cost_matrix(X,Y)
#    L3=ot.partial.partial_wasserstein
    L_pr=ot.partial.partial_wasserstein(mu,nu,M,mass)
    cost_pr=np.sum(M*L_pr)+(n-mass)*Lambda
    

    cost_lp,L_lp=opt_lp(X,Y,Lambda)
    mass_lp=np.sum(L_lp)
    cost_lp+=(n-mass_lp)*Lambda



    L_sinkhorn=ot.partial.entropic_partial_wasserstein(mu,nu,M,0.1,mass)
    cost_sinkhorn=np.sum(M*L_sinkhorn)+(n-mass)*Lambda
    
    cost_v2_list.append(cost_v2)
    cost_v2_a_list.append(cost_v2_a)
    cost_pr_list.append(cost_pr)
    cost_lp_list.append(cost_lp)
    cost_sinkhorn_list.append(cost_sinkhorn)
     
    

fig = plt.figure()
ax = plt.subplot(111)
plt.plot(range(start_n,end_n,step),cost_v2_list,'-',label='ours v2-apro')
plt.plot(range(start_n,end_n,step),cost_v2_a_list,label='ours v2')
plt.plot(range(start_n,end_n,step),cost_pr_list,label='Lp (primal): python OT')
plt.plot(range(start_n,end_n,step),cost_lp_list,label='Lp: python OT')
lt.plot(range(start_n,end_n,step),cost_lp_list,label='Sinkhorn: python OT')
plt.xlabel("n: size of X")
plt.ylabel("OPT distances")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=3)
plt.savefig('experiment/test/results/accuracy.jpg',dpi=fig.dpi,bbox_inches='tight')
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