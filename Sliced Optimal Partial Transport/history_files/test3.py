# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 17:49:22 2022

@author: laoba
"""
import math as mt
import cvxpy as cp
import numpy as np

import ot 


def cost(x,y,Lambda=1/4):#cost function c-lambda
    if abs(x-y)/(mt.sqrt(4*Lambda))>=mt.pi/2:
        return 100

    else:
        return -8*Lambda*mt.log(mt.cos(abs(x-y)/(mt.sqrt(4*Lambda))))



def entropy_function(t):
    if t==0:
        return 1 # the convension is 0*inf=0
    else:
        return t*mt.log(t)-t+1

    
def KL_divergence(Pi,mu):
    ''' plist is the relative density dpi_1/dmu 
        ''' 
    n=len(Pi)
    S=0
    for i in range(n):
        S+=entropy_function(Pi[i])*mu[i]
    return S

C=np.zeros(3)
x=[0,0.1,0.2]
y=[0]
Lambda=1/4
for i in range(3):
    C[i]=cost(x[i],y[0])

n=3
mu=np.ones(n) #The mass for each point is 1
m=1
nu=np.ones(m) #The mass for each point is 1

#minimize the Unbalanced OT by Convex optimization solver
Pi = cp.Variable(shape=3) #Pi is the joint distribution
zeros=np.zeros(3)
constraints=[Pi>=zeros] # The cosntraint that each entry is positive (or zero)
obj_F=C*Pi-cp.sum(cp.entr(Pi))-cp.sum(Pi)-cp.entr(cp.sum(Pi))-cp.sum(Pi)+4 # The object function of unbalanced OT. Note The entr(x) is xln(x).

obj=cp.Minimize(obj_F)
# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("Solve the problem by Convex problem minimizer cvxpy")
print("status:", prob.status)
print("optimal distance", prob.value)
print("optimal transporation plan", Pi.value)

# My solution 
Pi=[]# the density of P_1pi/mu 
for i in range(n):
    pd=mt.exp(-1/(4*Lambda)*cost(x[i],y[0]))
    Pi.append(pd)
Pi=np.array(Pi)/mt.sqrt(sum(Pi))
print("Solve the problem by hand")
print('My minimizer is')
print(Pi)
Total_cost=sum(Pi*C)+KL_divergence(Pi,mu)+KL_divergence([sum(Pi)],nu)
print('My total cost is')
print(Total_cost)



# Solution by POT 
M=np.zeros([n,m]) 
for i in range(n):
    for j in range(m):
        M[i,j]=cost(x[i],y[j])
        
reg=0.01
reg_m=np.floor(4*Lambda)
minimizer=ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, M, reg, reg_m)
print("Solve the problem by POT (Sinkhorn algorithm)")
print('the optimal tranportation plan is')
print(minimizer)
distance=ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M, reg, reg_m)
print('The transportation distance is '+str(distance))
 