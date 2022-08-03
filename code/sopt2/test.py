#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:34:44 2022

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
item='/stanford_bunny'
#'/stanford_bunny' #'/witchcastle' #'mumble_sitting'
exp_num='/stanford_bunny' 
#'/stanford_bunny'#'/witchcastle' #'mumble_sitting'

from sopt2.library import *
from sopt2.opt import *
#from sopt2.sliced_opt import *   

n=10
m=20
X=10*np.float32(np.random.rand(n))
Y=10*np.float32(np.random.rand(m))
X.sort()
Y.sort()
exp_point=np.float32(np.inf)
X1=np.append(X,exp_point)
mass_diff=m-n
mu=np.ones(n+1)
nu=np.ones(m+1)
mu[-1]=m
nu[-1]=n
Lambda=0.1
cost_M=np.zeros((n+1,m+1),dtype=np.float32)
cost_M1=cost_matrix(X1[0:-1],Y)
cost_M[0:n,0:m]=cost_M1-Lambda/2
plan_1=ot.lp.emd(mu,nu,cost_M)
plan=plan_1[0:n,0:m]
mass=np.sum(plan)

cost1=np.sum(cost_M1*plan)+(n-mass)*Lambda
cost2,L=opt_1d_v2(X,Y,Lambda)

            
    

