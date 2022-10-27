#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:16:14 2022

@author: baly
"""
import os
import numpy as np
from typing import Tuple
import torch
from scipy.stats import ortho_group
import sys
import numba as nb
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
from sopt2.library import *


#@nb.njit([nb.types.Tuple((nb.float32[:,:],nb.float32))(nb.float32[:,:],nb.float32[:,:])])
def recover_rotation_numba(X,Y):
    n,d=X.shape
    X_c=X-np.mean(X,0)
    Y_c=Y-np.mean(Y,0)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d)
    diag[d-1,d-1]=np.det(R.T)
    rotation=U.dot(diag).dot(VT)
    scaling=np.sum(np.abs(S.T))/np.trace(Y_c.T.dot(Y_c))
    return rotation,scaling

def recover_rotation_du_numba(X,Y):
    n,d=X.shape
    X_c=X-torch.mean(X,0)
    Y_c=Y-torch.mean(Y,0)
    YX=Y_c.T@X_c
    U,S,VT=torch.linalg.svd(YX)
    R=U@VT
    diag=torch.eye(d)
    diag[d-1,d-1]=torch.det(R)
    rotation=U@diag@VT
    E_list=torch.eye(3)
    scaling=torch.zeros(3)
    for i in range(3):
        Ei=torch.diag(E_list[i])
        num=0
        denum=0
        for j in range(3):
            num+=X_c[j].T@rotation.T@Ei@Y_c[j]
            denum+=Y_c[j].T@Ei@Y_c[j]
        scaling[i]=num/denum

    return rotation,scaling

