# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:14:35 2022

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


def rotation_matrix(theta):
    return torch.stack([
        torch.stack([torch.cos(theta),-torch.sin(theta)]),
        torch.stack([torch.sin(theta),torch.cos(theta)])
        ])
def rectangular(array1,array2):
    n=len(array1)
    m=len(array2)
    X0=torch.cat([d*torch.ones(m) for d in array1])
    X1=torch.cat([array2 for i in range(0,n)])
    return torch.tensor([X0.numpy(),X1.numpy()],device=device).T

        
    

diag_opt=torch.tensor([2.0,2.0],device=device)
theta_opt=torch.tensor(np.pi/3.0,device=device)
beta_opt=torch.tensor([3.0,3.0],device=device)
D_opt=torch.diag_embed(diag_opt) # optimal scaling matrix
ROT_opt=rotation_matrix(theta_opt) # optimal rotation matrix

diag=torch.tensor([1.0,1.0],device=device,requires_grad=True)
theta=torch.tensor(0.0,device=device,requires_grad=True)
beta=torch.tensor([0.0,0.0],device=device,requires_grad=True)


p=0.2
N=6
n=int(N*(1-p)) # of data points
n1=int(N*p) # of outliers
r2=5
r1=-5

data=torch.linspace(r1,r2,steps=n)
X=rectangular(data,data)
N_X=torch.tensor(np.random.uniform(r1-10,r1-10,size=[n1,2]),dtype=torch.float32,device=device)
X_hat=torch.cat([X,N_X])



Y=torch.matmul(torch.matmul(D_opt,ROT_opt),X.T).T+beta_opt
N_Y=torch.tensor(np.random.uniform(r2+15,r2+15,size=[n1,2]),dtype=torch.float32,device=device)
Y_hat=torch.cat([Y,N_Y])


#X_n=torch.rand
plt.scatter(X_hat.cpu()[:,0], X_hat.cpu()[:,1], c='red',label='X')
plt.scatter(Y_hat.cpu()[:,0], Y_hat.cpu()[:,1], c='blue',label='Y')

plt.legend()
plt.show()
nb_iter_max=300

lr1=1e-2
lr2=1e-3
lr3=1e-1
Lambda=5
beta_list=[]
theta_list=[]
diag_list=[]
optimizer=optim.Adam([diag,theta,beta],lr=0.5)

for i in range(0,nb_iter_max):   
    optimizer.zero_grad()
    D=torch.diag_embed(diag) 
    ROT=rotation_matrix(theta) 
    Y_hat_estimate =torch.matmul(torch.matmul(D,ROT),X_hat.T).T+beta
    loss=sopt(Y_hat,Y_hat_estimate,Lambda,n_projections=40)
    loss.backward()

    optimizer.step()
    if n%10==0:
        print('diag_grad',torch.norm(diag.grad))

    # performs a step of projected gradient descent




     
plt.scatter(Y_hat_estimate.clone().detach().cpu()[:,0], Y_hat_estimate.clone().detach().cpu()[:,1], c='red',label='Y_hat=DRX+beta')
plt.scatter(Y_hat.cpu()[:,0], Y_hat.cpu()[:,1], c='blue',label='Y')
plt.legend()
plt.show()        
