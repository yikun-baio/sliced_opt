#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu  
@author: Bernard Schmitzer, schmitzer @cs.uni-goettingen.de

"""


import numpy as np
import torch
import os
import sys
#import numba as nb
from typing import Tuple #,List
from numba.typed import List
import ot
import numba as nb 


import scipy
from ot.lp.emd_wrap import emd_c, check_result, emd_1d_sorted
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm


from .library import *


p=2  # global variable, the ground cost is (x-y)**p

#@nb.njit(nb.types.Tuple((nb.float64,nb.int64[:]))(nb.float64[:],nb.float64[:],nb.float64))
# solve opt by linear programming 
 
@nb.njit(cache=True)
def argmin_nb(X,Y):
    Min=np.inf
    ind=0
    m=Y.shape[0]
    for i in range(m):
        cost_xy=X[i]-Y[i]
        if cost_xy<Min:
            Min=cost_xy
            ind=i
    return ind


@nb.njit(cache=True)
def cost_function(x,y,p=2): 
    ''' 
    case 1:
        input:
            x: float number
            y: float number 
        output:
            (x-y)**2: float number 
    case 2: 
        input: 
            x: n*1 float np array
            y: n*1 float np array
        output:
            (x-y)**2 n*1 float np array, whose i-th entry is (x_i-y_i)**2
    '''
#    V=np.square(x-y) #**p
    V=np.abs(x-y)**p
    return V



# @nb.njit(['float64[:,:](float64[:])'],fastmath=True)
# def transpose(X):
#     n=X.shape[0]
#     XT=np.zeros((n,1),np.float64)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT

# @nb.njit(['float32[:,:](float32[:])'],fastmath=True)
# def transpose_32(X):
#     n=X.shape[0]
#     XT=np.zeros((n,1),np.float32)
#     for i in range(n):
#         XT[i]=X[i]
#     return XT


# @nb.njit(['float64[:,:](float64[:],float64[:])','float32[:,:](float32[:],float32[:])'],fastmath=True,cache=True)
# def cost_matrix(X,Y):
#     '''
#     input: 
#         X: (n,) float np array
#         Y: (m,) float np array
#     output:
#         M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
#     '''
#     XT=np.expand_dims(X,1)
#     M=cost_function(XT,Y)    
#     return M

@nb.njit(cache=True,fastmath=False,parallel=True)
def cost_matrix(X,Y,p=2):
    '''
    input: 
        X: (n,) float np array
        Y: (m,) float np array
    output:
        M: n*m matrix, M_ij=c(X_i,Y_j) where c is defined by cost_function.
    
    '''
#    XT=np.expand_dims(X,1)
    n,m=X.shape[0],Y.shape[0]
    M=np.zeros((n,m))
    for i in nb.prange(n):
        for j in nb.prange(m):
            M[i,j]=(X[i]-Y[j])**p   
    return M


@nb.njit(cache=True)
def argmin_nb(array):
    Min=np.inf
    Min_ind=0
    n=array.shape[0]
    for i in range(n):
        val=array[i]
        if val<Min:
            Min=val
            Min_ind=i
    return Min_ind,Min


    
def opt_lp(mu,nu,M,Lambda,numItermax=100000,numThreads=1):
    """
    Solves the partial optimal transport problem
    and returns the OT plan by linear programming in PythonOT 
    
    Parameters
    ----------
    mu : np.ndarray (dim_mu,) float64 
        Unnormalized histogram of dimension `dia_mu`
    nu : np.ndarray (dim_nu,) float64
        Unnormalized histograms of dimension `dia_nu`
    M : np.ndarray (dim_mu, dim_nu) float64
        cost matrix
    reg : float
        Regularization term > 0
    numItermax : int64, optional
        Max number of iterations


    Returns
    -------
    gamma : (dim_mu, dim_nu) ndarray
        Optimal transportation matrix for the given parameters
    cost : float64
        
    """
    n,m=M.shape 
    mu1=np.zeros(n+1)
    nu1=np.zeros(m+1)
    mu1[0:n]=mu
    nu1[0:m]=nu
    mu1[-1]=np.sum(nu)
    nu1[-1]=np.sum(mu)
    M1=np.zeros((n+1,m+1),dtype=np.float64)
    M1[0:n,0:m]=M-2*Lambda
    plan1, cost1, u, v, result_code = emd_c(mu1, nu1, M1, numItermax, numThreads)
    result_code_string = check_result(result_code)
    #plan1=ot.lp.emd(mu1,nu1,M1,numItermax=numItermax,numThreads=numThreads)
    plan=plan1[0:n,0:m]
    cost=np.sum(M*plan)
    return cost,plan

# 
@nb.njit((nb.float64[:,:])(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64,nb.float64,nb.int64),cache=True)
def sinkhorn_opt_pr(mu, nu, M, mass, reg, numItermax=100000):
    r"""
    (we modify the code in PythonOT) 
    Solves the partial optimal transport problem
    and returns the OT plan vis Sinkhorn algorithm (we modify the code in PythonOT)

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma,
                 \mathbf{M} \rangle_F + \mathrm{reg} \cdot\Omega(\gamma)

        s.t. \gamma \mathbf{1} &\leq \mathbf{a} \\
             \gamma^T \mathbf{1} &\leq \mathbf{b} \\
             \gamma &\geq 0 \\
             \mathbf{1}^T \gamma^T \mathbf{1} = m
             &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\} \\

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega`  is the entropic regularization term,
      :math:`\Omega=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[3] <references-entropic-partial-wasserstein>` (prop. 5)


    Parameters
    ----------
    mu : np.ndarray (dia_mu,) float64
        Unnormalized histogram of dimension `dia_mu`
    b : np.ndarray (dia_nu,) float64
        Unnormalized histograms of dimension `dia_nu`
    M : np.ndarray (dia_mu, dia_nu)
        cost matrix
    reg : float
        Regularization term > 0
    m : float64, optional
        Amount of mass to be transported
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (dia_mu, dia_nu) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> mu = [.1, .2]
    >>> nu = [.1, .1]
    >>> M = [[0., 1.], [2., 3.]]
    >>> np.round(entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    array([[0.06, 0.02],
           [0.01, 0.  ]])


    .. _references-entropic-partial-wasserstein:
    References
    ----------
    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
       (2015). Iterative Bregman projections for regularized transportation
       problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    See Also
    --------
    ot.partial.partial_wasserstein: exact Partial Wasserstein
    """

    #mu = np.asarray(a, dtype=np.float64)
    #nu = np.asarray(b, dtype=np.float64)
    #M = np.asarray(M, dtype=np.float64)

    dim_mu, dim_nu = M.shape
    dx = np.ones(dim_mu, dtype=np.float64)
    dy = np.ones(dim_nu, dtype=np.float64)
    stopThr=1e-13
            

    # Next 3 lines equivalent to K=np.exp(-M/reg), but faster to compute
    K=np.exp(-M/reg)
    #K = np.empty(M.shape, dtype=M.dtype)
    #np.divide(M, -reg, out=K)
    #np.exp(K, out=K)
    
    K=K*mass / np.sum(K) # make the total mass of K to be mass
    
    err, cpt = 1, 0
    q1 = np.ones(K.shape)
    q2 = np.ones(K.shape)
    q3 = np.ones(K.shape)

    while (err > stopThr and cpt < numItermax):
        Kprev = K
        K = K * q1
        K1 = np.dot(np.diag(np.minimum(mu / np.sum(K, axis=1), dx)), K)
        q1 = q1 * Kprev / K1
        K1prev = K1
        K1 = K1 * q2
        K2 = np.dot(K1, np.diag(np.minimum(nu / np.sum(K1, axis=0), dy)))
        q2 = q2 * K1prev / K2
        K2prev = K2
        K2 = K2 * q3
        K = K2 * (mass / np.sum(K2))
        q3 = q3 * K2prev / K


        if cpt % 10 == 0:
            err = np.linalg.norm(Kprev - K)

        cpt = cpt + 1
    if cpt==numItermax-1:
        print('warning, maximum iteration reached')
    # log_e['partial_w_dist'] = np.sum(M * K)
    # if log:
    #     return K, log_e
    return K

@nb.njit((nb.float64[:,:])(nb.float64[:],nb.float64[:],nb.float64[:,:],nb.float64,nb.int64),cache=True)
def sinkhorn_knopp(mu, nu, M, reg, numItermax=1000000):
    r"""
    we modify the code in PythonOT
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem via Sinkhorn Algorithm: (we modify the code in PythonOT)

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dia_mu`, `dia_nu`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    mu : array-like, shape (mu,) float64
        samples weights in the source domain
    nu : array-like, shape (nu,) float64
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed :math:`\mathbf{M}` if :math:`\mathbf{b}` is a matrix
        (return OT loss + dual variables in log)
    M : array-like, shape (dia_mu, dia_nu) float64
        loss matrix
    reg : float 
        Regularization term >0
    numItermax : int64, optional  
        Max number of iterations 
    stopThr : float64, optional 
        Stop threshold on error (>0)

    Returns
    -------
    gamma : array-like, shape (dim_mu, dim_nu)
        Optimal transportation matrix for the given parameters

    Examples
    --------

    >>> import ot
    >>> mu=[.5, .5]
    >>> nu=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    # init data
    dim_mu = mu.shape[0]
    dim_nu = nu.shape[0]
    stopThr=1e-9
    
    #initialize u,v 
    u = np.ones(dim_mu) # is exp()
    v = np.ones(dim_nu)

    K = np.exp(-M/reg)
    
    for ii in range(numItermax):
        u_pre=u.copy()
        v_pre=v.copy()
        v = nu / np.dot(K.T, u)
        u = mu / np.dot(K, v)
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(u_pre - u)+np.linalg.norm(v_pre - v)  # violation of marginal
            if err < stopThr:
                break
    if ii==numItermax-1:
        print('warning, maximum iteration reached')
    
    gamma=np.expand_dims(u,1)*(K*v.T)

    return gamma

@nb.njit(['(float64[:,:])(float64[:],float64[:],float64[:,:],float64,float64,int64)'],cache=True)
def sinkhorn_knopp_opt(mu, nu, M, Lambda, reg, numItermax=1000):
    r"""
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg}\cdot\Omega(\gamma)

        s.t. \ \gamma \mathbf{1} &= \mathbf{a}

             \gamma^T \mathbf{1} &= \mathbf{b}

             \gamma &\geq 0
    where :

    - :math:`\mathbf{M}` is the (`dia_mu`, `dia_nu`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term
      :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      weights (histograms, both sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp
    matrix scaling algorithm as proposed in :ref:`[2] <references-sinkhorn-knopp>`


    Parameters
    ----------
    mu : array-like, shape (mu,) float64
        samples weights in the source domain
    nu : array-like, shape (nu,) float64
    M : array-like, shape (dia_mu, dia_nu) float64
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int64, optional
        Max number of iterations
    stopThr : float64, optional
        Stop threshold on error (>0)

    Returns
    -------
    gamma : array-like, shape (dim_mu, dim_nu)
        Optimal transportation matrix for the given parameters

    Examples
    --------

    >>> import ot
    >>> mu=[.5, .5]
    >>> nu=[.5, .5]
    >>> M=[[0., 1.], [1., 0.]]
    >>> ot.sinkhorn(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    .. _references-sinkhorn-knopp:
    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
        of Optimal Transport, Advances in Neural Information
        Processing Systems (NIPS) 26, 2013


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """

    # init data
    dim_mu = mu.shape[0]
    dim_nu = nu.shape[0]
    stopThr=1e-9
    
    #initialize u,v 
    u = np.ones(dim_mu) # is exp()
    v = np.zeros(dim_nu)

    K = np.exp(-M/reg)
    
    for ii in range(numItermax):
        u_pre=u.copy()
        v_pre=v.copy()
        v = np.minimum(nu / np.dot(K.T, u),Lambda)
        u = np.minimum(mu / np.dot(K, v),Lambda)
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(u_pre - u)+np.linalg.norm(v_pre - v)  # violation of marginal
            if err < stopThr:
                break
    gamma=np.expand_dims(u,1)*(K*v.T)

    return gamma
    

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

    
@nb.njit(nb.types.Tuple((nb.float64,nb.float64[:],nb.float64[:],nb.int64[:],nb.int64[:]))(nb.float64[:,:],nb.float64),fastmath=True,cache=True)
def solve_opt(c,lam): #,verbose=False):
    M,N=c.shape
    phi=np.full(shape=M,fill_value=-np.inf)
    psi=np.full(shape=N,fill_value=lam)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=np.int64)
    piCol=np.full(N,-1,dtype=np.int64)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0
    # Dijkstra distance array, will be used and initialized on demand in case 3 subroutine
    dist=np.full(M,np.inf)

    jLast=-1
    while K<M:
#        if verbose: print(f"K={K}")
        if jLast==-1:
            j,val=argmin_nb(c[K,:]-psi)
        else:
            j,val=argmin_nb(c[K,jLast:]-psi[jLast:])
            j+=jLast
        #val=c[K,j]-psi[j]
        if val>=lam:
            #if verbose: print("case 1")
            phi[K]=lam
            K+=1
        elif piCol[j]==-1:
            #if verbose: print("case 2")
            piCol[j]=K
            piRow[K]=j
            phi[K]=val
            K+=1
            jLast=j
        else:
            #if verbose: print("case 3")
            phi[K]=val
            #assert piCol[j]==K-1
            # Dijkstra distance vector and currently explored radius
            dist[K]=0.
            dist[K-1]=0.
            v=0.

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
                    # catch: empty rows in between that could numerically be skipped
                    if iMin>0:
                        if piRow[iMin-1]==-1:
                            lowEndDiff=np.infty
                else:
                    lowEndDiff=np.infty
                # threshold for upper end
                if j<N-1:
                    hiEndDiff=c[K,j+1]-phi[K]-psi[j+1]-v
                else:
                    hiEndDiff=np.infty
                if hiEndDiff<=min((lowEndDiff,lamDiff)):
                 #  if verbose: print("case 3.2")
                    v+=hiEndDiff
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    
                    phi[K]+=v
                    piRow[K]=j+1
                    piCol[j+1]=K
                    jLast=j+1
                    resolved=True
                elif lowEndDiff<=min((hiEndDiff,lamDiff)):
                    if piCol[jMin-1]==-1:
                    #    if verbose: print("case 3.3a")
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
                        piRow[K]=j #jPrime
                        piCol[j]+=1 #jPrime
                        resolved=True
                    else:
                      #  if verbose: print("case 3.3b")
                      #  assert piCol[jMin-1]==iMin-1
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
                 #   if verbose: print(f"case 3.1, lamInd={lamInd}")
                    v+=lamDiff
                    
                    # domain=arange(iMin,K)
                    # phi[domain]+=v-dist[domain]
                    # psi[piRow[domain]]-=v-dist[domain]
                    # i=K-1
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    # "flip" assignment from lambda touching row onwards
                    if lamInd<K:
                        jPrime=piRow[lamInd]
                        piRow[lamInd]=-1
                        
                        # domain1=arange(lamInd+1,K)
                        # piRow[domain1]-=1
                        # piCol[domain1-(lamInd+1)+jPrime]+=1
                        # jPrime=K-(lamInd+1)+jPrime
                        # i=K-1
                        for i in range(lamInd+1,K):
                            piCol[jPrime]+=1
                            piRow[i]-=1
                            jPrime+=1
                        piRow[K]=j #jPrime
                        piCol[j]+=1 #jPrime
                    resolved=True
            #assert np.min(c-phi.reshape((M,1))-psi.reshape((1,N)))>=-1E-15
            K+=1
    objective=np.sum(phi)+np.sum(psi)
    return objective,phi,psi,piRow,piCol






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





@nb.njit(['Tuple((float64,int64[:]))(float64[:,:])'],cache=True)
def pot(M): 
    n,m=M.shape
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    cost=0.0 # save the optimal cost    
    argmin_Y=closest_y_M(M) # M.argmin(1)
 
    #initial loop:
    k=0
    #xk=X[k]
    jk=argmin_Y[k]
    cost_xk_yjk=M[k,jk]

    cost+=cost_xk_yjk
    L=np.append(L,jk)
    for k in range(1,n):
        jk=argmin_Y[k]
        cost_xk_yjk=M[k,jk]
        j_last=L[-1]
    
        #define consistent term     
        if jk>j_last:# No conflict, L[-1] is the j last assig
            cost+=cost_xk_yjk
            L=np.append(L,jk)
        else:
            # this is the case for conflict: 

            # compute the first cost 
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost1=cost+cost_xk_yjlast1
            else:
                cost1=np.inf 
            # compute the second cost 
            i_act,j_act=unassign_y(L)
            if j_act>=0:                        
                L1=np.concatenate((L[0:i_act],np.array([j_act]),L[i_act:]))
                X_indices=arange(0,k+1)
                Y_indices=L1
#                Y_assign=Y[L1]
#                X_assign=X[0:k+1]
                cost2=np.sum(matrix_take(M,X_indices,Y_indices))
#                cost2=np.sum(cost_function(X_assign,Y_assign))
            else:
                cost2=np.inf
            if cost1<cost2:
                cost=cost1
                L=np.append(L,j_last+1)
            elif cost2<=cost1:
                cost=cost2
                L=L1.copy()    
    return cost,L



