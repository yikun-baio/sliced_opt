#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:19:13 2022

@author: baly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:11:19 2022

@author: baly
"""


import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.rc('image', interpolation='nearest')
matplotlib.rc('figure',facecolor='white')
matplotlib.rc('image',cmap='viridis')

# %matplotlib inline

# +
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


# +
# this is more or less verbatim the version described in the notes
def solve1DPOT(c,lam,verbose=False,plots=False):
    M,N=c.shape
    
    phi=np.full(shape=M,fill_value=-np.inf)
    psi=np.full(shape=N,fill_value=lam)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=int)
    piCol=np.full(N,-1,dtype=int)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0

    while K<M:
        if verbose: print(f"K={K}")
        j=np.argmin(c[K,:]-psi)
        val=c[K,j]-psi[j]
        if val>=lam:
            if verbose: print("case 1")
            phi[K]=lam
            K+=1
        elif piCol[j]==-1:
            if verbose: print("case 2")
            piCol[j]=K
            piRow[K]=j
            phi[K]=val
            K+=1
        else:
            if verbose: print("case 3")
            phi[K]=val
            #assert piCol[j]==K-1
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
                    hiEndDiff=c[K,j+1]-phi[K]-psi[j+1]
                else:
                    hiEndDiff=np.infty
                if hiEndDiff<=min(lowEndDiff,lamDiff):
                    if verbose: print("case 3.1")
                    phi[iMin:K+1]+=hiEndDiff
                    psi[jMin:j+1]-=hiEndDiff
                    piRow[K]=j+1
                    piCol[j+1]=K
                    resolved=True
                elif lowEndDiff<=min(hiEndDiff,lamDiff):
                    if piCol[jMin-1]==-1:
                        if verbose: print("case 3.2a")
                        phi[iMin:K+1]+=lowEndDiff
                        psi[jMin:j+1]-=lowEndDiff
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
                        if verbose: print("case 3.2b")
                        assert piCol[jMin-1]==iMin-1
                        phi[iMin:K+1]+=lowEndDiff
                        psi[jMin:j+1]-=lowEndDiff
                        # adjust distance to threshold
                        lamDiff-=lowEndDiff
                        iMin-=1
                        jMin-=1
                        if lam-phi[iMin]<lamDiff:
                            lamDiff=lam-phi[iMin]
                            lamInd=iMin

                else:
                    if verbose: print(f"case 3.3, lamInd={lamInd}")
                    phi[iMin:K+1]+=lamDiff
                    psi[jMin:j+1]-=lamDiff
                    # "flip" assignment from lambda touching row onwards
                    jPrime=piRow[lamInd]
                    piRow[lamInd]=-1
                    for i in range(lamInd+1,K):
                        piCol[jPrime]+=1
                        piRow[i]-=1
                        jPrime+=1
                    if lamInd<K:
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                    resolved=True
                #assert np.min(c-phi.reshape((M,1))-psi.reshape((1,N)))>=-1E-15
            K+=1
        if plots:
            fig=plt.figure(figsize=(12,4))
            fig.add_subplot(1,3,1)
            plt.title(f"K={K-1}")
            cEff=c-phi.reshape((M,1))-psi.reshape((1,N))
            plt.imshow(cEff<=1E-15)
            fig.add_subplot(1,3,2)
            plt.imshow(getPiFromRow(M,N,piRow))
            fig.add_subplot(1,3,3)
            plt.imshow(getPiFromCol(M,N,piCol))
            plt.show()
    return phi,psi,piRow,piCol


# in this version the adjustment of dual variables is not done throughout subroutine 3
# but only upon its conclusion, thus one internal loop over the entries of duals is skipped
# this version may have worst case complexity O(N^2)
def solve1DPOTDijkstra(c,lam,verbose=False,plots=False):
    M,N=c.shape
    
    phi=np.full(shape=M,fill_value=-np.inf)
    psi=np.full(shape=N,fill_value=lam)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=int)
    piCol=np.full(N,-1,dtype=int)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0

    while K<M:
        if verbose: print(f"K={K}")
        j=np.argmin(c[K,:]-psi)
        val=c[K,j]-psi[j]
        if val>=lam:
            if verbose: print("case 1")
            phi[K]=lam
            K+=1
        elif piCol[j]==-1:
            if verbose: print("case 2")
            piCol[j]=K
            piRow[K]=j
            phi[K]=val
            K+=1
        else:
            if verbose: print("case 3")
            phi[K]=val
            #assert piCol[j]==K-1
            # Dijkstra distance vector and currently explored radius
            dist=np.full(M,np.inf)
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
                    if verbose: print("case 3.1")
                    v+=hiEndDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    piRow[K]=j+1
                    piCol[j+1]=K
                    resolved=True
                elif lowEndDiff<=min(hiEndDiff,lamDiff):
                    if piCol[jMin-1]==-1:
                        if verbose: print("case 3.2a")
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
                        if verbose: print("case 3.2b")
                        assert piCol[jMin-1]==iMin-1
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
                    if verbose: print(f"case 3.3, lamInd={lamInd}")
                    v+=lamDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    # "flip" assignment from lambda touching row onwards
                    jPrime=piRow[lamInd]
                    piRow[lamInd]=-1
                    for i in range(lamInd+1,K):
                        piCol[jPrime]+=1
                        piRow[i]-=1
                        jPrime+=1
                    if lamInd<K:
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                    resolved=True
            #assert np.min(c-phi.reshape((M,1))-psi.reshape((1,N)))>=-1E-15
            K+=1
        if plots:
            fig=plt.figure(figsize=(12,4))
            fig.add_subplot(1,3,1)
            plt.title(f"K={K-1}")
            cEff=c-phi.reshape((M,1))-psi.reshape((1,N))
            plt.imshow(cEff<=1E-15)
            fig.add_subplot(1,3,2)
            plt.imshow(getPiFromRow(M,N,piRow))
            fig.add_subplot(1,3,3)
            plt.imshow(getPiFromCol(M,N,piCol))
            plt.show()
    return phi,psi,piRow,piCol

# -

M=4
N=4
n=M
m=N
lam=30.0
#x=np.random.random(size=M)
#y=np.random.random(size=N)

X=np.random.uniform(-20,20,n)
# np.random.seed(m)
Y=np.random.uniform(-40,40,m)
#X.sort()
#Y.sort()
#x=np.random.normal(size=M)
#y=2*np.random.normal(size=N)-0.5
#x=np.sort(X)
#y=np.sort(Y)
c=getCost(X,Y)

plt.plot(X,marker="x")
plt.plot(Y,marker="x")
plt.show()


phi,psi,piRow,piCol=solve1DPOT(c,lam,verbose=False,plots=False)
phiD,psiD,piRowD,piColD=solve1DPOTDijkstra(c,lam,verbose=False,plots=False)

print(np.sum(np.abs(phi-phiD)))
print(np.sum(np.abs(psi-psiD)))
print(np.sum(np.abs(piRow-piRowD)))
print(np.sum(np.abs(piCol-piColD)))

cEff=c-phi.reshape((M,1))-psi.reshape((1,N))
print(np.min(cEff))
pi=getPiFromRow(M,N,piRow)
print(np.sum(np.abs(pi-getPiFromCol(M,N,piCol))))

fig=plt.figure(figsize=(12,4))
fig.add_subplot(1,2,1)
plt.imshow(cEff<=1E-15)
fig.add_subplot(1,2,2)
plt.imshow(pi)
plt.show()
print('cost is',np.sum(pi*c))

X.sort()
Y.sort()
c=getCost(X,Y)

plt.plot(X,marker="x")
plt.plot(Y,marker="x")
plt.show()


phi,psi,piRow,piCol=solve1DPOT(c,lam,verbose=False,plots=False)
phiD,psiD,piRowD,piColD=solve1DPOTDijkstra(c,lam,verbose=False,plots=False)

print(np.sum(np.abs(phi-phiD)))
print(np.sum(np.abs(psi-psiD)))
print(np.sum(np.abs(piRow-piRowD)))
print(np.sum(np.abs(piCol-piColD)))

cEff=c-phi.reshape((M,1))-psi.reshape((1,N))
print(np.min(cEff))
pi=getPiFromRow(M,N,piRow)
print(np.sum(np.abs(pi-getPiFromCol(M,N,piCol))))

fig=plt.figure(figsize=(12,4))
fig.add_subplot(1,2,1)
plt.imshow(cEff<=1E-15)
fig.add_subplot(1,2,2)
plt.imshow(pi)
plt.show()
print('cost is',np.sum(pi*c))

# np.sum(pi*cEff)

# print(np.sum(np.abs(phi[piRow==-1]-lam)))
# print(np.sum(np.abs(psi[piCol==-1]-lam)))




