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
        return 2000

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
y=[0,100,101]
n=3
m=1
C=np.zeros([n,m])

Lambda=1/4
for i in range(n):
    for j in range(m):
        C[i,j]=cost(x[i],y[j])

mu=np.ones(n) #The mass for each point is 1
nu=np.ones(m) #The mass for each point is 1
print('The problem is:')
print('mu is '+str(mu)) 
print('on points'+str(x))
print('nu is'+str(nu))
print('on points'+str(y[0]))
print('============================')

#minimize the Unbalanced OT by Convex optimization solver
Pi = cp.Variable([n,m]) #Pi is the joint distribution
zeros_1=np.zeros([n])
zeros_2=np.zeros([n,m-1])
ones_1=np.ones(m)
ones_2=np.ones(n)
constraints=[Pi[:,0]>=zeros_1] # The cosntraint that each entry is positive (or zero)
obj_F=sum(sum(cp.multiply(C,Pi)))+sum(cp.kl_div(sum(Pi.T),mu))+sum(cp.kl_div(sum(Pi),nu)) 
# The object function of unbalanced OT. Note The entr(x) is -xln(x).

obj=cp.Minimize(obj_F)
# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("Solve the problem by Convex problem minimizer cvxpy")
#print("status:", prob.status)
print("optimal distance (with KL divergence term)", prob.value)
print("optimal distance (without KL divergence term)", sum(sum(Pi.value*C)))
print("optimal transporation plan is")
print(Pi.value)
print('=================================')
# My solution 
Pi=[]# the density of P_1pi/mu 
for i in range(n):
    pd=mt.exp(-1/(4*Lambda)*cost(x[i],y[0]))
    Pi.append(pd)
Pi=np.array(Pi)/mt.sqrt(sum(Pi))
print('solve the problem by hand')
total_cost=sum(Pi*C[:,0])+KL_divergence(Pi,mu)+KL_divergence([sum(Pi)],nu)
print('my distance (with KL divergence) is',total_cost)
print('my distance (without KL divergence) is',np.dot(Pi,C[:,0]))
print('My minimizer is')
print(Pi)

print('=================================')





reg=0.01
reg_m=4*Lambda
Pi=ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, C, reg, reg_m)
distance=ot.unbalanced.sinkhorn_unbalanced2(mu, nu, C, reg, reg_m)
total_cost=sum(sum(Pi*C))+KL_divergence(np.dot(Pi,ones_1),mu)+KL_divergence(np.dot(Pi.T,ones_2),nu)

print('Solve the problem by POT package (Sinkhorn algorithm)')

print('The transportation distance (with KL divergence) is',total_cost)

print('The transportation distance (without KL divergence) is',distance[0])

print('the optimal tranportation plan is')
print(Pi)


print('=================================')


 