#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:52:30 2022

@author: baly
"""


import numpy as np
import torch
import os
import sys
import numba as nb
from typing import Tuple #,List
from numba.typed import List
import ot
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


from sopt2.library import *

#@nb.njit(nb.types.Tuple((nb.float32,nb.int64[:]))(nb.float32[:],nb.float32[:],nb.float32))
def opt_lp(X,Y,Lambda,numItermax=100000):
    n=X.shape[0]
    m=Y.shape[0]
    exp_point=np.float32(np.inf)
    X1=np.append(X,exp_point)
    Y1=np.append(Y,exp_point)
    mu1=np.ones(n+1)
    nu1=np.ones(m+1)
    mu1[-1]=m
    nu1[-1]=n
    cost_M=cost_matrix(X1[0:-1],Y1[0:-1])
    cost_M1=np.zeros((n+1,m+1),dtype=np.float32)
    cost_M1[0:n,0:m]=cost_M-Lambda
    plan1=ot.lp.emd(mu1,nu1,cost_M1,numItermax=numItermax)
    plan=plan1[0:n,0:m]
    cost=np.sum(cost_M*plan)
    return cost,plan

    

    