#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 20:16:47 2022

@author: baly
"""

data=torch.load('data.pt')
X=data['X']
Y=data['Y']
n=X.shape[0]
m=Y.shape[0]
M=cost_matrix(X,Y)
cost2,L_v2=opt_1d_v2(X,Y,Lambda)
L_v2=plan_to_matrix(L_v2,m)
cost_v2=np.sum(M*L_v2)+Lambda*np.sum(n-np.sum(L_v2))+1/2*Lambda*(m-n)

cost_lp,L_lp=opt_lp(X,Y,Lambda)
mass_lp=np.sum(L_lp)
cost_lp=np.sum(M*L_lp)+Lambda*(n-mass_lp)+1/2*Lambda*(m-n)
print(abs(cost_v2-cost_lp))




b=-10
X=X0 #[1:] #[:b]
Y=Y0 #[5:] #[:-10]
X.sort()
Y.sort()
Lambda=100.0
n=X.shape[0]
m=Y.shape[0]
M=cost_matrix(X,Y)
cost2,L_v2=opt_1d_v2(X,Y,Lambda)
L_v2=plan_to_matrix(L_v2,m)
cost_v2=np.sum(M*L_v2)+Lambda*np.sum(n-np.sum(L_v2))+1/2*Lambda*(m-n)

cost_lp,L_lp=opt_lp(X,Y,Lambda)
mass_lp=np.sum(L_lp)
cost_lp=np.sum(M*L_lp)+Lambda*(n-mass_lp)+1/2*Lambda*(m-n)
print(cost_v2-cost_lp)

L_v2=matrix_to_plan(L_v2)
L_lp=matrix_to_plan(L_lp)
print(np.where(L_v2-L_lp>0)[0][0])

data={}
data['X']=X
data['Y']=Y
torch.save(data,'data.pt')
X0=X.copy()
Y0=Y.copy()