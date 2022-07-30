#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 07:09:48 2022
@author: baly
"""

import sys
import open3d as o3d
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import ot

work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)

from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *

def plan_to_matrix(L,m):
    n=L.shape[0]
    Lx=np.arange(n)
    Lx=Lx[L>=0]
    M=np.zeros((n,m))
    for i in Lx:
        M[i,L[i]]=1
    return M

def domain_of_plan(plan):
    n,m=plan.shape
    domain=[]
    for i in range(n):
        if np.sum(plan[i]>0)>0:
            domain.append(i)
    domain=np.array(domain)
    return domain

for i in range(10):
    n=20
    s=8
    X=torch.rand(n,2)
    Y=torch.rand(n,2)+1
    Lambda_list=torch.linspace(1,3,20)[1:]
    A=sopt_correspondence(X,Y,Lambda_list,400)
    A.get_plans_lambda()
    domain1=A.get_high_frequency_plan(s)

    
    mass=s #torch.sum(A.Lx_max>=0).item()
    a=np.ones(n)
    b=np.ones(n)
    Cost_M=cost_matrix_d(X.numpy(),Y.numpy())
    M2=ot.partial.partial_wasserstein(a, b, Cost_M, m=mass, nb_dummies=1)
    domain2=domain_of_plan(M2)
    if np.linalg.norm(domain1-domain2)>0:
        print('X',X)
        print('Y',Y)
        break


    
    
    