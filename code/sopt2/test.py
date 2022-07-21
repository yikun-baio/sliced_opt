#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 08:36:46 2022

@author: baly
"""

import numpy as np
import math
import torch
import os

import numba as nb 
#from numba.types import Tuple
from typing import Tuple
import sys
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.library import *

X=np.array([-5,  2,3,4],dtype=np.float32)
Y=np.array([-10,1,2,3,7],dtype=np.float32)
M=cost_matrix(X,Y)
n,m=M.shape
argmin_Y=closest_y_M(M)
Occupy=np.zeros(m,dtype=np.int32)


k=0
jk=argmin_Y[0]
free_y=jk-1

X_list=(np.array([k]),)
Y_list=(np.array([jk]),)
free_Y=np.array([free_y])

Occupy[jk]=1 # 
Range=arange(1,n)
for k in Range:
    jk=argmin_Y[k]
    cxy=M[k,jk]
        
    if Occupy[jk]==0:
        free_y=jk-1
        if Occupy[free_y]==1:
            free_y=free_Y[-1]
        X_list+=(np.array([k]),)
        Y_list+=(np.array([jk]),)
        free_Y=np.concatenate((free_Y,np.array([free_y])))
        Occupy[jk]=1


    elif Occupy[jk]==1: # If it is occupied, we should extend the current problem
        last_free_y=free_Y[-1] # it must be occupied by last problem 
        n=free_Y.shape[0]
        index1=np.where(free_Y==last_free_y)[0][0]
        
        # if last problem is adjacent to previous problems, merge all of them
        if index1<n-1: #from problem index1 to the last one, we need to merge them
            merged_X=np.concatenate(X_list[index1:])
            merged_Y=np.concatenate(Y_list[index1:])
            X_list=X_list[0:index1]+(merged_X,)
            Y_list=Y_list[0:index1]+(merged_Y,)
            free_Y=free_Y[0:index1+1]

        
        #extend the last problem      
        last_X=X_list[-1] # the last problem must have jk
        last_Y=Y_list[-1]
        last_y=last_Y[-1]
        last_free_y=free_Y[-1]
        X_list=X_list[0:-1]+(np.concatenate((last_X,np.array([k]))),)
        if last_free_y>=0:
            last_Y=np.concatenate((np.array([last_free_y]),last_Y))
            Occupy[last_free_y]=1
        if last_y+1<=m-1:
            last_Y=np.concatenate((last_Y,np.array([last_y+1])))
            Occupy[last_y+1]=1
        Y_list=Y_list[0:-1]+(last_Y,)
            
        free_y=last_Y[0]-1
        if free_y>=0 and Occupy[free_y]==1:
            free_y=free_Y[-2]
        free_Y[-1]=free_y
        
            
            
        
        
        
        
            
        
            
            
            
                    
        
    
    
    
@nb.njit()
def test(x):
    C=List()
    C.append(-2)
    C.append(-2)
    A=tuple(C)
 #   for i in range(3):
 #       C=C+(1,)
        
#    C=tuple(L)
 

    return C
    
    

    