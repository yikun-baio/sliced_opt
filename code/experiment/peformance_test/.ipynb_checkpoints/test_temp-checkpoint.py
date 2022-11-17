#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 19:47:07 2022

@author: baly
"""

data=torch.load('data.pt')
X0=data['X0']
Y0=data['Y0']

L_v2=matrix_to_plan(L_v2)
L_lp=matrix_to_plan(L_lp)
print(np.where(L_v2-L_lp>0))

X0=X.copy()
Y0=Y.copy()
data={}
data['X0']=X0
data['Y0']=Y0
data['lambda']=Lambda

torch.save(data,'data.pt')



a=0
b=0
Lambda=np.float32(10)
X=X0[:].astype(np.float64)
Y=Y0[:].astype(np.float64) #np.delete(Y0,[655, 656, 657, 658, 659])
n=X.shape[0]
m=Y.shape[0]
M=cost_matrix(X,Y)


cost_v2,L_v2=opt_1d_v2(X,Y,Lambda)
L_v2=plan_to_matrix(L_v2,m)
cost_v2=np.sum(M*L_v2)+Lambda*np.sum(n-np.sum(L_v2))

#   cost_v2-=Lambda*np.sum(n-np.sum(L_v2>=0))


mu=np.ones(n)
nu=np.ones(m)    
cost_lp,L_lp=opt_lp(X,Y,Lambda)
mass_lp=np.sum(L_lp)
cost_lp=np.sum(M*L_lp)+Lambda*(n-mass_lp)

print(cost_v2-cost_lp)
print('n is',n)
L_v2=matrix_to_plan(L_v2)
L_lp=matrix_to_plan(L_lp)
print(np.where(L_v2-L_lp>0))

L =np.arange(75)
i_start=0
j_start=0
j_last=L[-1]
k=75
M1=M[i_start:k,j_start:j_last]
L1=L[i_start:k].copy()

#     # we need the last assign index since we need to retrieve the closest unassigend j                    

cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub(M1,L1,Lambda)

X=X[i_start:k].astype(np.float64)
Y=Y[j_start:j_last].astype(np.float64)
M1=cost_matrix(X,Y)
X1=np.delete(X,1)
C1=np.sum(cost_function(X1,Y))

i=4
C41=np.sum(cost_list[i:i+s])
