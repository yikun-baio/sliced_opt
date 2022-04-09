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
        return 10000

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


x=[0,0.1,0.2]
y=[0]
Lambda=1/4

n=3
mu=np.ones(n) #The mass for each point is 1 for mu[1,1,1]
m=1
nu=np.ones(m) #The mass for each point is 1 for nu [1]

C=np.zeros([n,m])#Define cost matrix 
for i in range(n):
    for j in range(m):
        C[i,j]=cost(x[i],y[j])




#minimize the Unbalanced OT by Convex optimization solver
Pi = cp.Variable(shape=[n,m]) #Pi is the joint distribution
zeros=np.zeros([n,m])
constraints=[Pi>=zeros] # The cosntraint that each entry is positive (or zero)
trans_cost=C.T*Pi
KL_1=-cp.sum(cp.entr(Pi))-cp.sum(Pi)+sum(mu)
KL_2=-cp.entr(sum(Pi))-cp.sum(Pi)+sum(nu)
obj_F=trans_cost+KL_1+KL_2 # The object function of unbalanced OT. Note The entr(x) is xln(x).

obj=cp.Minimize(obj_F)
# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("Solve the problem by cvxpy")
#print("status:", prob.status)
print("optimal transporation plan is")
print(Pi.value.T)
trans_cost=sum((C.T*Pi.value)[0])
print("transportation cost without KL divergence is ", trans_cost)
print("transportation cost with KL divergence is ", prob.value)

print("====================================================")

# My solution 
Pi=[]# the density of Pi/mu 
for i in range(n):
    pd=mt.exp(-1/(4*Lambda)*cost(x[i],y[0]))
    Pi.append(pd)
Pi=np.array(Pi)/mt.sqrt(sum(Pi))
print("Solve the problem by hand")
print('My minimizer is')
print(Pi)
Trans_cost=sum((C.T*Pi)[0])
print('transportation cost without KL divergence is',Trans_cost)

Total_cost=Trans_cost+KL_divergence(Pi,mu)+KL_divergence([sum(Pi)],nu)
print('transportation cost with KL divergence is',Total_cost)

print("====================================================")

# Solution by POT 

        
reg=0.001
reg_m=np.floor(4*Lambda)
minimizer=ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, C, reg, reg_m)
print("Solve the problem by POT (Sinkhorn algorithm)")
print('the optimal tranportation plan is')
print(minimizer.T)
trans_cost=ot.unbalanced.sinkhorn_unbalanced2(mu, nu, C, reg, reg_m)[0]
print('The transportation cost (without KL divergence) is '+str(trans_cost))
print("====================================================")
print("The End")
 
