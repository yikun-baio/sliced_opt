# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:20:18 2022

@author: laoba
"""

import torch
import numpy as np

import ot 
import matplotlib.pyplot as plt
import time
import os

parent_path='G:/My Drive/Github/Yikun-Bai/Sliced-Optimal-Partial-Transport/Code'

dataset_path =parent_path+'/datasets'
Image_path = dataset_path+'/Images/task1'

lab_path=parent_path+'/sopt'
#
os.chdir(lab_path)
 
from multiprocessing import Pool

n=5
X=1+5*torch.rand([n,3]).requires_grad_()
Y=torch.rand([n+3,3])*10-3 
Lambda=4
n_projections=3

from pooling import *

if __name__ == '__main__':
    cost1=sopt_pool(X,Y,Lambda,n_projections)
    cost2=sopt(X,Y,Lambda,n_projections)
    print(cost1)
    print(cost2)


    start_n=10
    end_n=200

    step=10
    k=1
    Lambda=8
    time2_list=[]
    time3_list=[]
    for n in range (start_n,end_n,step):
        m=n+10
        mu=np.ones(n)
        nu=np.ones(m)
        time1=0
        time2=0
        time3=0
        time4=0
        time5=0
        for i in range (k):
            X=np.random.uniform(0,20,size=n)
            Y=np.random.uniform(5,30,size=m)
        

    
        

    
        start_time = time.time()
        X.sort()
        Y.sort()    
        X_torch=torch.tensor(X,device='cuda',requires_grad=True)
        Y_torch=torch.tensor(Y,device='cuda')
        cost3,L3=opt_1d_v2(X_torch,Y_torch,Lambda)
        end_time = time.time()
        time3+=end_time-start_time
    
    time2_list.append(time2/k)
    time3_list.append(time3/k)

plt.plot(range(start_n,end_n,step),time2_list,label='ours v2 numpy')
plt.plot(range(start_n,end_n,step),time3_list,label='ours v2 pytorch')

#plt.plot(range(start_n,end_n,step),time5_list,label='LP: python ot')
plt.xlabel("n: size of X")
plt.ylabel("runing time")
plt.legend(loc='best')
plt.show()
