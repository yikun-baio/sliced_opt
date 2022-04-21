# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:57:58 2022

@author: laoba
"""

import torch
import numpy as np 
from OPT import *
from library import *
import ot 
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
#device='cpu'

theta_0_opt=5.0
theta_1_opt=2.0

theta_0_torch=torch.tensor(0.0,device=device,requires_grad=True)
theta_1_torch=torch.tensor(1.0,device=device,requires_grad=True)

n=18
X=np.random.uniform(0,10,n)
Y=theta_0_opt+theta_1_opt*X
N_x=np.random.uniform(-15,-5,2)
N_y=np.random.uniform(30,40,2)
X_hat=np.concatenate((X,N_x))
Y_hat=np.concatenate((Y,N_y))

X_torch = torch.tensor(X_hat).to(device=device)
Y_torch = torch.tensor(Y_hat).to(device=device)


X_torch=X_torch.sort().values
Y_torch=Y_torch.sort().values

error_theta_0_list=[]
error_theta_1_list=[]



nb_iter_max = 600

Lambda=10
loss_iter = []
theta_0_list = []
theta_1_list=[]
for i in range(nb_iter_max):
    Y_hat_torch =theta_0_torch+theta_1_torch*X_torch
    Y_hat_torch=Y_hat_torch.sort().values
    loss,L1=OPT_1D_v3(Y_hat_torch,Y_torch,Lambda)
    loss.backward()

    # performs a step of projected gradient descent
    with torch.no_grad():
        lr1 =5e-2
        lr2= 1e-4
        #print(grad.item())
        theta_0_torch -= theta_0_torch.grad *lr1 # step
        theta_1_torch -= theta_1_torch.grad *lr2
        theta_0_torch.grad.zero_()
        theta_1_torch.grad.zero_()
        error1=abs(theta_0_opt-theta_0_torch.clone().detach().cpu().numpy())
        error2=abs(theta_1_opt-theta_1_torch.clone().detach().cpu().numpy())
        error_theta_0_list.append(error1)
        error_theta_1_list.append(error2)
     
        

#plt.plot(range(0,nb_iter_max),error_theta_0_list,label='theta_0')
#plt.plot([0,nb_iter_max],[theta_0_opt,theta_0_opt],label='optimal theta_0')
plt.semilogy(range(0,nb_iter_max),error_theta_0_list,label='error of theta_0')
plt.semilogy(range(0,nb_iter_max),error_theta_1_list,label='error of theta_1')
plt.xlabel("# of iteration")
plt.ylabel("error")
plt.title("find the optimal shift and scalar")
plt.legend()
plt.show()        