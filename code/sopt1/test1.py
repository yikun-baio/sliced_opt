# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import torch.multiprocessing as mp
import torch
import time
import numpy as np
import sys
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
device='cpu'
from sopt2.library import *
X=np.array([0.1,0.2,0.3],dtype=np.float32)
Y=np.array([0.1,0.2,0.6,0.6],dtype=np.float32)
M=cost_matrix(X,Y)

n,m=M.shape
#Lambda=np.float32(Lambda)
 
L=np.empty(0,dtype=np.int32) # save the optimal plan
cost=np.float32(0) # save the optimal cost

argmin_Y=closest_y_M(M)
Y_occupy=np.zeros(4,dtype=np.int32)
L=[]

    
def problem_merge(P_list):
    P={}
    P['X']=np.concatenate([P['X'] for P in P_list])
    P['Y']=np.concatenate([P['Y'] for P in P_list])
    P['start']=P_list[0]['start']
    P['end']=P_list[-1]['end']
    return P
    
for i in range(3):
    j=argmin_Y[i]
    if Y_occupy[j]==0:# not yet be used
        Y_occupy[j]=1
        P={}
        P['X']=X[i:i+1]
        P['Y']=Y[j:j+1]
        if j-1>=0 and Y_occupy[j-1]==1:
            P_last=L[-1]
            start_last=last_problem['start']
            start=start_last
        else:
            start=j-1
        problem['start']=start
        problem['end']=j+1
        L.append(problem)
   # elif Y_occupy[j]==1 and j==argmin_Y[i-1]:
        
        
        
#    elif Y_occupy[j]==1 and argmin_Y[i]==argmin_Y[i-1]: # it is used
        
        
        
        
        
    
        
    




    
    
    
