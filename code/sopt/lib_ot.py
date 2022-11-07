#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:52:30 2022

@author: baly
"""


import numpy as np
import torch
import os
import sys
#import numba as nb
from typing import Tuple #,List
from numba.typed import List
import ot
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
import numba as nb 


import scipy
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm


from sopt.library import *

#@nb.njit(nb.types.Tuple((nb.float64,nb.int64[:]))(nb.float64[:],nb.float64[:],nb.float64))
# solve opt by linear programming 
def opt_lp(X,Y,Lambda,numItermax=100000):
    n=X.shape[0]
    m=Y.shape[0]
    exp_point=np.float64(np.inf)
    X1=np.append(X,exp_point)
    Y1=np.append(Y,exp_point)
    mu1=np.ones(n+1)
    nu1=np.ones(m+1)
    mu1[-1]=m
    nu1[-1]=n
    cost_M=cost_matrix(X1[0:-1],Y1[0:-1])
    cost_M1=np.zeros((n+1,m+1),dtype=np.float64)
    cost_M1[0:n,0:m]=cost_M-Lambda
    plan1=ot.lp.emd(mu1,nu1,cost_M1,numItermax=numItermax)
    plan=plan1[0:n,0:m]
    cost=np.sum(cost_M*plan)
    return cost,plan

    

import numpy as np

def getCost(x,y,p=2.):
    """Squared Euclidean distance cost for two 1d arrays"""
    c=(x.reshape((-1,1))-y.reshape((1,-1)))
    c=c**p
    return c

def getPiFromRow(M,N,piRow):
    pi=np.zeros(shape=(M,N),dtype=int)
    for i,j in zip(np.arange(M),piRow):
        if j>-1: pi[i,j]=1
    return pi

def getPiFromCol(M,N,piCol):
    pi=np.zeros(shape=(M,N),dtype=int)
    for i,j in zip(piCol,np.arange(N)):
        if i>-1: pi[i,j]=1
    return pi


@nb.njit(nb.types.Tuple((nb.float64,nb.float64[:],nb.float64[:],nb.int64[:],nb.int64[:]))(nb.float64[:,:],nb.float64))
def solve_opt(c,lam): #,verbose=False):
    M,N=c.shape
    
    phi=np.full(shape=M,fill_value=-np.inf)
    psi=np.full(shape=N,fill_value=lam)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=np.int64)#int)
    piCol=np.full(N,-1,dtype=np.int64)#int)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0
    # Dijkstra distance array, will be used and initialized on demand in case 3 subroutine
    dist=np.full(M,np.inf)

    jLast=-1
    while K<M:
#        if verbose: print(f"K={K}")
        if jLast==-1:
            j=np.argmin(c[K,:]-psi)
        else:
            j=jLast+np.argmin(c[K,jLast:]-psi[jLast:])
        val=c[K,j]-psi[j]
        if val>=lam:
 #           if verbose: print("case 1")
            phi[K]=lam
            K+=1
        elif piCol[j]==-1:
  #          if verbose: print("case 2")
            piCol[j]=K
            piRow[K]=j
            phi[K]=val
            K+=1
            jLast=j
        else:
   #         if verbose: print("case 3")
            phi[K]=val
    #        assert piCol[j]==K-1
            # Dijkstra distance vector and currently explored radius
            dist[K]=0.
            dist[K-1]=0.
            v=0

            # iMin and jMin indicate lower end of range of contiguous rows and cols
            # that are currently examined in subroutine;
            # upper end is always K and j
            iMin=K-1
            jMin=j
            # threshold until an entry of phi hits lam
            if phi[K]>phi[K-1]:
                lamDiff=lam-phi[K]
                lamInd=K
            else:
                lamDiff=lam-phi[K-1]
                lamInd=K-1
            resolved=False
            while not resolved:
                # threshold until constr iMin,jMin-1 becomes active
                if jMin>0:
                    lowEndDiff=c[iMin,jMin-1]-phi[iMin]-psi[jMin-1]
                else:
                    lowEndDiff=np.infty
                # threshold for upper end
                if j<N-1:
                    hiEndDiff=c[K,j+1]-phi[K]-psi[j+1]-v
                else:
                    hiEndDiff=np.infty
                if hiEndDiff<=min(lowEndDiff,lamDiff):
     #               if verbose: print("case 3.1")
                    v+=hiEndDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    piRow[K]=j+1
                    piCol[j+1]=K
                    jLast=j+1
                    resolved=True
                elif lowEndDiff<=min(hiEndDiff,lamDiff):
                    if piCol[jMin-1]==-1:
      #                  if verbose: print("case 3.2a")
                        v+=lowEndDiff
                        for i in range(iMin,K):
                            phi[i]+=v-dist[i]
                            psi[piRow[i]]-=v-dist[i]
                        phi[K]+=v
                        # "flip" assignment along whole chain
                        jPrime=jMin
                        piCol[jMin-1]=iMin
                        piRow[iMin]-=1
                        for i in range(iMin+1,K):
                            piCol[jPrime]+=1
                            piRow[i]-=1
                            jPrime+=1
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                        resolved=True
                    else:
       #                 if verbose: print("case 3.2b")
                    #    assert piCol[jMin-1]==iMin-1
                        v+=lowEndDiff
                        dist[iMin-1]=v
                        # adjust distance to threshold
                        lamDiff-=lowEndDiff
                        iMin-=1
                        jMin-=1
                        if lam-phi[iMin]<lamDiff:
                            lamDiff=lam-phi[iMin]
                            lamInd=iMin

                else:
        #            if verbose: print(f"case 3.3, lamInd={lamInd}")
                    v+=lamDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    # "flip" assignment from lambda touching row onwards
                    if lamInd<K:
                        jPrime=piRow[lamInd]
                        piRow[lamInd]=-1
                        for i in range(lamInd+1,K):
                            piCol[jPrime]+=1
                            piRow[i]-=1
                            jPrime+=1
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                    resolved=True
            #assert np.min(c-phi.reshape((M,1))-psi.reshape((1,N)))>=-1E-15
            K+=1
    objective=np.sum(phi)+np.sum(psi)
    return objective,phi,psi,piRow,piCol



@nb.njit(nb.types.Tuple((nb.float32,nb.float32[:],nb.float32[:],nb.int64[:],nb.int64[:]))(nb.float32[:,:],nb.float32))
def solve_opt_32(c,lam): #,verbose=False):
    M,N=c.shape
    
    phi=np.full(shape=M,fill_value=-np.inf,dtype=np.float32)
    psi=np.full(shape=N,fill_value=lam,dtype=np.float32)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=np.int64)#int)
    piCol=np.full(N,-1,dtype=np.int64)#int)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0
    # Dijkstra distance array, will be used and initialized on demand in case 3 subroutine
    dist=np.full(M,np.inf,dtype=nb.float32)

    jLast=-1
    while K<M:
#        if verbose: print(f"K={K}")
        if jLast==-1:
            j=np.argmin(c[K,:]-psi)
        else:
            j=jLast+np.argmin(c[K,jLast:]-psi[jLast:])
        val=c[K,j]-psi[j]
        if val>=lam:
 #           if verbose: print("case 1")
            phi[K]=lam
            K+=1
        elif piCol[j]==-1:
  #          if verbose: print("case 2")
            piCol[j]=K
            piRow[K]=j
            phi[K]=val
            K+=1
            jLast=j
        else:
   #         if verbose: print("case 3")
            phi[K]=val
    #        assert piCol[j]==K-1
            # Dijkstra distance vector and currently explored radius
            dist[K]=0.
            dist[K-1]=0.
            v=0

            # iMin and jMin indicate lower end of range of contiguous rows and cols
            # that are currently examined in subroutine;
            # upper end is always K and j
            iMin=K-1
            jMin=j
            # threshold until an entry of phi hits lam
            if phi[K]>phi[K-1]:
                lamDiff=lam-phi[K]
                lamInd=K
            else:
                lamDiff=lam-phi[K-1]
                lamInd=K-1
            resolved=False
            while not resolved:
                # threshold until constr iMin,jMin-1 becomes active
                if jMin>0:
                    lowEndDiff=c[iMin,jMin-1]-phi[iMin]-psi[jMin-1]
                else:
                    lowEndDiff=np.infty
                # threshold for upper end
                if j<N-1:
                    hiEndDiff=c[K,j+1]-phi[K]-psi[j+1]-v
                else:
                    hiEndDiff=np.infty
                if hiEndDiff<=min(lowEndDiff,lamDiff):
     #               if verbose: print("case 3.1")
                    v+=hiEndDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    piRow[K]=j+1
                    piCol[j+1]=K
                    jLast=j+1
                    resolved=True
                elif lowEndDiff<=min(hiEndDiff,lamDiff):
                    if piCol[jMin-1]==-1:
      #                  if verbose: print("case 3.2a")
                        v+=lowEndDiff
                        for i in range(iMin,K):
                            phi[i]+=v-dist[i]
                            psi[piRow[i]]-=v-dist[i]
                        phi[K]+=v
                        # "flip" assignment along whole chain
                        jPrime=jMin
                        piCol[jMin-1]=iMin
                        piRow[iMin]-=1
                        for i in range(iMin+1,K):
                            piCol[jPrime]+=1
                            piRow[i]-=1
                            jPrime+=1
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                        resolved=True
                    else:
       #                 if verbose: print("case 3.2b")
                    #    assert piCol[jMin-1]==iMin-1
                        v+=lowEndDiff
                        dist[iMin-1]=v
                        # adjust distance to threshold
                        lamDiff-=lowEndDiff
                        iMin-=1
                        jMin-=1
                        if lam-phi[iMin]<lamDiff:
                            lamDiff=lam-phi[iMin]
                            lamInd=iMin

                else:
        #            if verbose: print(f"case 3.3, lamInd={lamInd}")
                    v+=lamDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    # "flip" assignment from lambda touching row onwards
                    if lamInd<K:
                        jPrime=piRow[lamInd]
                        piRow[lamInd]=-1
                        for i in range(lamInd+1,K):
                            piCol[jPrime]+=1
                            piRow[i]-=1
                            jPrime+=1
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                    resolved=True
            #assert np.min(c-phi.reshape((M,1))-psi.reshape((1,N)))>=-1E-15
            K+=1
    objective=np.sum(phi)+np.sum(psi)
    return objective,phi,psi,piRow,piCol






    