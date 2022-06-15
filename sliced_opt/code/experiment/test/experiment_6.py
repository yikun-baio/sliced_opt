# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:07:42 2022

@author: laoba
"""

import numpy as np
import math
import ot
#from ot.sliced import get_random_projections
import os

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
lab_path=parent_path+'\\sopt'
os.chdir(lab_path)

from library import *
from opt import *
from sliced_opt import *
import matplotlib.pyplot as plt



n=40
m=40
d=4
X=torch.normal(0,1,size=[n,d])
Y=torch.normal(0.5,1.5,size=[m,d])
mu=torch.ones(n)
nu=torch.ones(m)

cost1_list=[]
cost2_list=[]
Lambda_list=np.linspace(0,10,10)
for Lambda in Lambda_list:
    cost1=ot.sliced.sliced_wasserstein_distance(X,Y,mu,nu,n_projections=1000,p=2)
    cost1=cost1**2
    cost2=sliced_opt(X,Y,Lambda,1000)
    cost1_list.append(cost1)
    cost2_list.append(cost2)

plt.plot(Lambda_list,cost1_list,label='sliced OT')
plt.plot(Lambda_list,cost2_list,label='sliced OPT')

plt.xlabel("lambda")
plt.ylabel("Distance")
plt.title("sliced OPT and sliced OT")
plt.legend(loc='best')
plt.show()

    
    
