# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:48:46 2022

@author: laoba
"""

import torch
import numpy as np 
import sys
import os
import matplotlib.pyplot as plt

work_path=os.path.dirname(__file__)

loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
root='experiment/dataset_similarity/data'
from sopt2.lib_set import *


from sopt2.opt import *
from sopt2.library import *




# Experiment 1 Relation between OT and OPT
n=25
m=30
X=np.float32(np.random.uniform(0,10,size=n))
Y=np.float32(np.random.uniform(3,15,size=m))
X.sort()
Y.sort()
mu=np.ones(n)
nu=np.ones(m)

cost_matrix=cost_matrix(X,Y)
##cost_matrix=np.zeros([n,m])
for i in range(n):
    for j in range(m):
        cost_matrix[i,j]=cost_function(X[i],Y[j])
cost1,L1=pot_1d(X,Y)

Lambda_list=np.linspace(0,40,60)
cost2_list=[]
for Lambda in Lambda_list:    
    cost2,L2=opt_1d_v1(X,Y,Lambda)
    cost2_list.append(cost2)
    
plt.plot(Lambda_list,cost2_list,label='OPT distance v1')

plt.plot([Lambda_list[0],Lambda_list[-1]],[cost1,cost1],label='POT distance')
plt.xlabel("lambda")
plt.ylabel("Distance")
plt.title("OPT and POT")
plt.legend()
plt.show()