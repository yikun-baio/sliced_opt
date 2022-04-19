# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:29:08 2022

@author: laoba
"""
import numpy as np 
import math
import ot



def cost(x,y,Lambda=1/4):#cost function c-lambda
    if abs(x-y)/(np.sqrt(4*Lambda))>=np.pi/2:
        return np.inf

    else:
        return -8*Lambda*np.log(np.cos(abs(x-y)/(np.sqrt(4*Lambda))))

def entropy_function(t):
    if t==0:
        return 1 # the convension is 0*inf=0
    else:
        return t*np.log(t)-t+1

    
def KL_divergence(Pi,mu):
    ''' plist is the relative density dpi_1/dmu 
        ''' 
    n=len(Pi)
    S=0
    for i in range(n):
        S+=entropy_function(Pi[i])*mu[i]
    return S



n=3
m=3
mu=1/3*np.ones(n)
nu=1/3*np.ones(m)
Lambda=1/4

x=[0,0.1,0.2]
y=[0,100,101]
print('Problem 2')
print('mu is',mu) 
print('defined on points',x)
print('nu is',nu) 
print('defined on points',y[0:m])
print('============================')

C=np.zeros([n,m])
ones_1=np.ones(m)
ones_2=np.ones(n)

Lambda=1/4
for i in range(n):
    for j in range(m):
        C[i,j]=cost(x[i],y[j])
        

#minimize the Unbalanced OT by Convex optimization solver



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




Pi=ot.sinkhorn(mu, nu, C, reg)
#distance=ot.unbalanced.sinkhorn_unbalanced2(mu, nu, C, reg, reg_m)
#total_cost=sum(sum(Pi*C))+KL_divergence(np.dot(Pi,ones_1),mu)+KL_divergence(np.dot(Pi.T,ones_2),nu)
print('Solve the original OT problem by POT package (Sinkhorn algorithm)')
#print('The transportation distance (with KL divergence) is',total_cost)
#print('The transportation distance (without KL divergence) is',distance[0])
print('the optimal tranportation plan is')
print(Pi)

