# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:52:13 2022

@author: laoba
"""

import torch
import numpy as np 
from OPT import *
from library import *
import ot 
import matplotlib.pyplot as plt


# Experiment 1 Relation between OT and OPT
n=20
m=20
X=np.random.uniform(0,10,size=n)
Y=np.random.uniform(5,15,size=m)
X.sort()
Y.sort()
mu=np.ones(n)
nu=np.ones(m)

M=cost_matrix(X,Y)

cost1=ot.emd2(mu,nu,M)

Lambda_list=np.linspace(0,80,10)
cost2_list=[]
for Lambda in Lambda_list:    
    cost2,L2=OPT_1D_v1(X,Y,Lambda)
    cost2_list.append(cost2)
    
plt.plot(Lambda_list,cost2_list,label='OPT distance v1')

plt.plot([Lambda_list[0],Lambda_list[-1]],[cost1,cost1],label='OT distance (POT package)')
plt.xlabel("lambda")
plt.ylabel("Distance")
plt.title("OPT and OT")
plt.legend()
plt.show()



# dtype = torch.float
# device = torch.device("cuda")
# theta = torch.randn((), device=device, dtype=dtype, requires_grad=True)
# X=np.random.rand(10)*3
# noise1=np.random.rand(5)-5
# noise2=np.random.rand(5)+5

# X1=np.concatenate((X,noise1))
# Y1=np.concatenate((X,noise2))





# device = "cuda" if torch.cuda.is_available() else "cpu"
# #device='cpu'
# theta=torch.tensor([3.0],device=device,requires_grad=True)
# X1.sort()
# Y1.sort()

# X0_torch = torch.tensor(X1).to(device=device)
# Y0_torch = torch.tensor(Y1).to(device=device)


# nb_iter_max = 230

# theta_all = np.zeros([nb_iter_max, theta.shape[0]])
# loss_iter = []



# for i in range(nb_iter_max):
#     Y1_torch = Y0_torch+theta
#     loss,L1=OPT_v2(X0_torch,Y1_torch)
#     loss.backward()

#     # performs a step of projected gradient descent
#     with torch.no_grad():
#         lr =5e-2
#         grad = theta.grad
#         #print(grad.item())
#         theta -= grad * lr # step
#         theta.grad.zero_()
#         theta_all[i,:] = theta.clone().detach().cpu().numpy()







