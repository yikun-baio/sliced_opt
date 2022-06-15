# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:26:46 2022

@author: laoba
"""


import numpy as np
import math
import torch 
import os
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
lab_path=parent_path+'\\sopt'
os.chdir(lab_path)



import torch
import numpy as np 
from opt import *
from library import *
from sliced_opt import *

import ot 
import matplotlib.pyplot as plt
import time
import torch.optim as optim

#device = "cuda" if torch.cuda.is_available() else "cpu"
device='cpu'

        
    
n=2
d=3

X=torch.tensor([[0.0,0.0]],requires_grad=True)

Y=torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]])*1.0


#X_n=torch.rand
plt.scatter(X.clone().detach().cpu()[:,0], X.clone().detach().cpu()[:,1], c='red',label='X')
plt.scatter(Y.cpu()[:,0], Y.cpu()[:,1], c='blue',label='Y')
plt.legend()

plt.show()
nb_iter_max=200

X_list=[]
Lambda=400

optimizer=optim.Adam([X],lr=0.001)


for i in range(1000):
    i=i+1
    optimizer.zero_grad()
    loss=sopt_es(X,Y,Lambda,n_projections=30)
    loss.backward()
    optimizer.step()
    if i%100==0:
        print('grad',torch.norm(X.grad))
        print('loss',loss)
    if loss<0.1:
        break



plt.scatter(X.clone().detach().cpu()[:,0], X.clone().detach().cpu()[:,1], c='red',label='Y_hat=DRX+beta')
plt.scatter(Y.cpu()[:,0], Y.cpu()[:,1], c='blue',label='Y')
plt.legend()
plt.show()        