# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:48:46 2022

@author: laoba
"""

import torch
import numpy as np 
import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
lab_path=parent_path+'\\SOPT'

from lab_path import *
# from library import *
# import ot 
# import matplotlib.pyplot as plt








# # Experiment 1 Relation between OT and OPT
# n=18
# m=25
# X=np.random.uniform(0,10,size=n)
# Y=np.random.uniform(3,15,size=m)
# X.sort()
# Y.sort()
# mu=np.ones(n)
# nu=np.ones(m)


# cost_matrix=np.zeros([n,m])
# for i in range(n):
#     for j in range(m):
#         cost_matrix[i,j]=cost_function(X[i],Y[j])
# cost1,L1=POT_1D(X,Y)

# Lambda_list=np.linspace(0,40,60)
# cost2_list=[]
# for Lambda in Lambda_list:    
#     cost2,L2=OPT_1D_v1(X,Y,Lambda)
#     cost2_list.append(cost2)
    
# plt.plot(Lambda_list,cost2_list,label='OPT distance v1')

# plt.plot([Lambda_list[0],Lambda_list[-1]],[cost1,cost1],label='POT distance')
# plt.xlabel("lambda")
# plt.ylabel("Distance")
# plt.title("OPT and POT")
# plt.legend()
# plt.show()