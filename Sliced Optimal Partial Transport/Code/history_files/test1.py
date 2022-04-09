# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:59:55 2022

@author: laoba
"""

import numpy as np
from math import *

import ot 


def cost(x,y): #cost function c-lambda
    if abs(x-y)/(sqrt(4*Lambda))>=pi/2:
        return 100

    else:
        return -8*Lambda*log(cos(abs(x-y)/(sqrt(4*Lambda))))


def entropy_function(t):
    if t==0:
        return 1 # the convension is 0*inf=0
    else:
        return t*log(t)-t+1

    
def KL_divergence(pid,mu):
    ''' plist is the relative density dpi_1/dmu 
        ''' 
    n=len(pid)
    S=0
    for i in range(n):
        S+=entropy_function(pid[i])*mu[i]
    return S

def T(xi):
    if xi in xl:
        return yl[0]
    else:
        return None
    
def T_prime(xi):
    if xi==xl[0]:
        return yl[0]
    else:
        return None    

#

n=3
Lambda=1/4 

mu=1/n*np.ones(n) # mu is the pmf of each point
xl=[0,0.1,0.2]


m=1
k=1 # mass
nu=k/n*np.ones([m]) # nu is the pmf of each point
yl=[0,200,201]

pi1d=[]# the density of P_1pi/mu 
pi1d_sum=0


for i in range(n):
    pd=exp(-1/(4*Lambda)*cost(xl[i],yl[0]))
    pi1d_sum+=pd
    pi1d.append(pd)
pi1d_sum=sqrt(pi1d_sum)
pi1d=np.array(pi1d)/pi1d_sum

pi2d=np.zeros(m) # the density of p_2pi/nu
pi2d[0]=sum(pi1d)

pi1d_prime=np.zeros(n) # for T', we just transfer all the mass from x1 to y1. 
pi1d_prime[0]=1

pi2d_prime=np.zeros(m)
pi2d_prime[0]=1
# Now we compute the total cost induced by pi 
trans_cost_pi=0 # the transportation cost that transfer mass from x1,x2,x3 to y1
for i in range(n):
    trans_cost_pi+=cost(xl[i],yl[0])*pi1d[i]*mu[i]
KL_mu1_pi=KL_divergence(pi1d,mu)
KL_mu2_pi=KL_divergence(pi2d,nu) #For the second term, we only consider

Total_cost_pi=trans_cost_pi+4*Lambda*KL_mu1_pi+4*Lambda*KL_mu2_pi  
print('The total cost calculated by mapping T is '+str(Total_cost_pi))


total_cost=1+m/3-2/3*sqrt(cos(abs(xl[0]-yl[0]))**2+cos(abs(xl[1]-yl[0]))**2+cos(abs(xl[2]-yl[0]))**2)

print('The total cost calculated by hand is '+str(total_cost))


# Now we compute the total cost induced by pi'
trans_cost_pi_prime=0 # the transportation cost that transfer mass from x1 to y1
for i in range(n):
    trans_cost_pi_prime+=cost(xl[i],yl[0])*pi1d_prime[i]*mu[i]
KL_mu1_pi_prime=KL_divergence(pi1d_prime,mu)
KL_mu2_pi_prime=KL_divergence(pi2d_prime,nu) #For the second term, we only consider

Total_cost_pi_prime=trans_cost_pi_prime+4*Lambda*KL_mu1_pi_prime+4*Lambda*KL_mu2_pi_prime  
print('The total cost calculated by 1-1 mapping T-prime is '+str(Total_cost_pi_prime))


# now we try it by Sinkhorn unbalanced function. 

M=np.zeros([n,m]) 
for i in range(n):
    for j in range(m):
        M[i,j]=cost(xl[i],yl[j])



reg=0.01
reg_m=np.floor(4*Lambda)

minimizer=ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, M, reg, reg_m)
print('the optimal tranportation plan is')
print(minimizer)
print ('my minimizer is')
print(k*pi1d/n) 

distance=ot.unbalanced.sinkhorn_unbalanced2(mu, nu, M, reg, reg_m)
print('The transportation distance is '+str(distance))
print('My transportation distance is '+str(Total_cost_pi))

a=1  
b=cos(0.2)**2
c=cos(0.3)**2
sumabc=sqrt(a+b+c)
A1=1+1/3
A2=-1/3*log(a)*a/sqrt(a+b+c)-1/3*log(b)*b/sqrt(a+b+c)-1/3*log(c)*c/sqrt(a+b+c)
A3=+1/3*a/sqrt(a+b+c)*log(a/sqrt(a+b+c))+1/3*b/sqrt(a+b+c)*log(b/sqrt(a+b+c))+1/3*c/sqrt(a+b+c)*log(c/sqrt(a+b+c))
A6=-1/3*a/sqrt(a+b+c)-1/3*b/sqrt(a+b+c)-1/3*c/sqrt(a+b+c)
A4=+1/3*sqrt(a+b+c)*log(sqrt(a+b+c))
A5=-1/3*sqrt(a+b+c)
print(A1+A2+A3+A4+A5+A6)