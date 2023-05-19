

@nb.njit((nb.float32[:,:])(nb.float32[:],nb.float32[:],nb.float32[:,:],nb.float32,nb.int64),cache=True)
def sinkhorn_knopp_32(mu, nu, M, reg, numItermax=1000000):
    r"""
    we modify the code in PythonOT
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
    mu : array-like, shape (mu,) float32
        samples weights in the source domain
    nu : array-like, shape (nu,) float32
    M : array-like, shape (dia_mu, dia_nu) float32
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int64, optional 
        Max number of iterations 
    stopThr : float32, optional 
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
    -------

    """

    # init data
    dim_mu = mu.shape[0]
    dim_nu = nu.shape[0]
    stopThr=1e-9
    
    #initialize u,v 
    u = np.ones(dim_mu,dtype=np.float32) # is exp()
    v = np.ones(dim_nu,dtype=np.float32)

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


@nb.njit(['(float32[:,:])(float32[:],float32[:],float32[:,:],float32,float32,int64)'],cache=True)
def sinkhorn_knopp_opt_32(mu, nu, M, Lambda, reg, numItermax=1000000):
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
        samples in the target domain, 
    M : array-like, shape (dia_mu, dia_nu)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int64, optional
        Max number of iterations
    stopThr : float32, optional
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
    stopThr=np.float32(1e-9)
    
    #initialize u,v 
    u = np.ones(dim_mu,dtype=np.float32) # is exp()
    v = np.zeros(dim_nu,dtype=np.float32)

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






@nb.njit(nb.types.Tuple((nb.float32,nb.float32[:],nb.float32[:],nb.int64[:],nb.int64[:]))(nb.float32[:,:],nb.float32),cache=True)
def solve_opt_32(c,lam): #,verbose=False):
    M,N=c.shape
    
    phi=np.full(shape=M,fill_value=-np.inf,dtype=np.float32)
    psi=np.full(shape=N,fill_value=lam,dtype=np.float32)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=np.int64)
    piCol=np.full(N,-1,dtype=np.int64)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0
    # Dijkstra distance array, will be used and initialized on demand in case 3 subroutine
    dist=np.full(M,np.inf,dtype=np.float32)

    jLast=-1
    while K<M:
#        if verbose: print(f"K={K}")
        if jLast==-1:
            j=np.argmin(c[K,:]-psi)
        else:
            j=jLast+np.argmin(c[K,jLast:]-psi[jLast:])
        val=c[K,j]-psi[j]
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
            dist[K]=np.float32(0)
            dist[K-1]=np.float32(0)
            v=np.float32(0)

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
                            lowEndDiff=np.float32(np.infty)
                else:
                    lowEndDiff=np.float32(np.infty)
                # threshold for upper end
                if j<N-1:
                    hiEndDiff=c[K,j+1]-phi[K]-psi[j+1]-v
                else:
                    hiEndDiff=np.float32(np.infty)
                if hiEndDiff<=min(lowEndDiff,lamDiff):
                 #  if verbose: print("case 3.1")
                    v+=hiEndDiff
                    # domain1=arange(iMin,K)
                    # phi[domain1]+=v-dist[domain1]
                    # psi[piRow[domain1]]-=v-dist[domain1]
                    # i=K-1
                    
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
                    #    if verbose: print("case 3.3a")
                        v+=lowEndDiff
                        # domain1=arange(iMin,K)
                        # phi[domain1]+=v-dist[domain1]
                        # psi[piRow[domain1]]-=v-dist[domain1]
                        # i=K-1
                        for i in range(iMin,K):
                            phi[i]+=v-dist[i]
                            psi[piRow[i]]-=v-dist[i]
                        phi[K]+=v
                        # "flip" assignment along whole chain
                        jPrime=jMin
                        piCol[jMin-1]=iMin
                        piRow[iMin]-=1
                        # domain2=arange(iMin+1,K)
                        # piCol[domain2-(iMin+1)+jPrime]+=1
                        # piRow[domain2]-=1
                        # jPrime=K-(iMin+1)+jPrime
                        # i=K-1
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
                    for i in range(iMin,K):
                        phi[i]+=v-dist[i]
                        psi[piRow[i]]-=v-dist[i]
                    phi[K]+=v
                    # "flip" assignment from lambda touching row onwards
                    if lamInd<K:
                        jPrime=piRow[lamInd]
                        piRow[lamInd]=-1
                        domain1=arange(lamInd+1,K)
                        piRow[domain1]-=1
                        piCol[domain1-(lamInd+1)+jPrime]+=1
                        jPrime=K-(lamInd+1)+jPrime
                        i=K-1
                        # for i in range(lamInd+1,K):
                        #     piCol[jPrime]+=1
                        #     piRow[i]-=1
                        #     jPrime+=1
                        piRow[K]=jPrime
                        piCol[jPrime]+=1
                    resolved=True
            #assert np.min(c-phi.reshape((M,1))-psi.reshape((1,N)))>=-1E-15
            K+=1
    objective=np.sum(phi)+np.sum(psi)
    #print('done')
    return objective,phi,psi,piRow,piCol



@nb.njit()
def solve(c,lam,verbose=False): # Dr. Barnhard
    M,N=c.shape
    
    phi=np.full(shape=M,fill_value=-np.inf)
    psi=np.full(shape=N,fill_value=lam)
    # to which cols/rows are rows/cols currently assigned? -1: unassigned
    piRow=np.full(M,-1,dtype=int)
    piCol=np.full(N,-1,dtype=int)
    # a bit shifted from notes. K is index of the row that we are currently processing
    K=0
    # Dijkstra distance array, will be used and initialized on demand in case 3 subroutine
    dist=np.full(M,np.inf)

    jLast=-1
    while K<M:
        if verbose: print(f"K={K}")
        if jLast==-1:
            j=np.argmin(c[K,:]-psi)
        else:
            j=jLast+np.argmin(c[K,jLast:]-psi[jLast:])
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
            jLast=j
        else:
            if verbose: print("case 3")
            phi[K]=val
            assert piCol[j]==K-1
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
                if hiEndDiff<=min(lowEndDiff,lamDiff):
                    if verbose: print("case 3.1")
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



@nb.njit(['Tuple((float32,int64[:]))(float32[:,:])'],cache=True)
def pot_32(M): 
    n,m=M.shape 
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    cost=np.float32(0)  # save the optimal cost
    argmin_Y=closest_y_M(M) #M.argmin(1)
 
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

                cost2=np.sum(matrix_take(M,X_indices,Y_indices))
#                cost2=np.sum(cost_function(X_assign,Y_assign))
            else:
                cost2=np.float32(np.inf)
            if cost1<cost2:
                cost=cost1
                L=np.append(L,j_last+1)
            elif cost2<=cost1:
                cost=cost2
                L=L1.copy()    
    return cost,L



@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],int64)'],fastmath=True,cache=True)
def transform_32(Xs0,Xsc,Xs,batch_size=128):    
    
    # # perform out of sample mapping
    n,m=Xs0.shape
    indices = np.arange(0,Xs0.shape[0])
    batch_ind = [indices[i:i + batch_size] for i in np.arange(0, len(indices), batch_size)]
    transp_Xs = np.zeros((n,m),dtype=np.float32)
    #transp_Xs=[]
    for bi in batch_ind:
        # get the nearest neighbor in the source domain
        D0 = cost_matrix_d(Xs0[bi], Xsc)
        idx = np.argmin(D0, axis=1)
        # define the transported points
        transp_Xs[bi] =Xs0[bi]+Xs[idx, :]  - Xsc[idx, :]
#        transp_Xs_=Xs0[bi]+Xs1[idx, :]  - Xs[idx, :]
        #print(transp_Xs)
#        transp_Xs.append(transp_Xs_)
#    transp_Xs = np.concatenate(transp_Xs, axis=0)
    return transp_Xs



@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],float32[:,:],int64)'],cache=True)
def spot_transfer_32(Xs0,Xt0,Xs,Xt,n_projections=400):
    n,d=Xs.shape
    #np.random.seed(0)
    projections=random_projections_32(d,n_projections,1)
    Xsc=Xs.copy()
    X_correspondence_pot_32(Xs,Xt,projections)     
    batch_size=128
    transp_Xs=transform_32(Xs0,Xsc,Xs,batch_size)
    return transp_Xs



@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:],int64)'],cache=True)
def sopt_transfer_32(Xs0,Xt0,Xs,Xt,Lambda_list,n_projections=400):    
    n,d=Xs.shape
    #np.random.seed(0)
    projections=random_projections_32(d,n_projections,1)
    Xsc=Xs.copy()
    X_correspondence_32(Xs,Xt,projections,Lambda_list)     
    batch_size=128
    transp_Xs=transform_32(Xs0,Xsc,Xs,batch_size)
    return transp_Xs


# OT-based color adaptation 
def ot_transfer_32(Xs0,Xt0,Xs,Xt,numItermax=1000000):
    n,d=Xs.shape
    m=Xt.shape[0]
    #plan=ot.emd()
    # get the transporation plan
    Xsc=Xs.copy()
    M=cost_matrix_d(Xs,Xt)
    mu=np.ones(n,dtype=np.float32)/n
    nu=np.ones(m,dtype=np.float32)/m
    plan=ot.lp.emd(mu, nu, M, numItermax=numItermax)

    # get the transported Xs It is the barycentric projection of Xt (with respect to Xs) 
    cond_plan=plan/np.expand_dims(np.sum(plan,1),1)
    Xs=np.dot(cond_plan,Xt)
    
#    # # prediction between images (using out of sample prediction as in [6])
    batch_size=128
    transp_Xs = transform_32(Xs0,Xsc,Xs,batch_size)
    return transp_Xs



@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,int64)'],cache=True)
def eot_transfer_32(Xs0,Xt0,Xs,Xt,reg=0.1,numItermax=1000000):
    n,d=Xs.shape
    m=Xt.shape[0]
    #plan=ot.emd()
    # get the transporation plan
    Xsc=Xs.copy()
    M=cost_matrix_d(Xs,Xt)
    mu=np.ones(n,dtype=np.float32)/n
    nu=np.ones(m,dtype=np.float32)/m
    plan=sinkhorn_knopp_32(mu, nu, M, reg=reg,numItermax=numItermax)

    # get the transported Xs
    cond_plan=plan/np.expand_dims(np.sum(plan,1),1)
    Xs=np.dot(cond_plan,Xt)
    
#    # # prediction between images (using out of sample prediction as in [6])
    batch_size=128
    transp_Xs = transform_32(Xs0,Xsc,Xs,batch_size)
    
    return transp_Xs


@nb.njit(['float32[:](float32[:,:])'],fastmath=True,cache=True)
def vec_mean_32(X):
    """
    return X.mean(1) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot32
    
    Return:
    --------
    mean: numpy array, shape (d,), float32 
    
    
    """

        
    n,d=X.shape
    mean=np.zeros(d,dtype=np.float32)
    for i in nb.prange(d):
        mean[i]=X[:,i].mean()
    return mean



@nb.njit(['Tuple((float32[:,:],float32))(float32[:,:],float32[:,:])'],cache=True)
def recover_rotation_32(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot32, target
    Y: numpy array, shape (n,d), flaot32, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float32 
    scaling: float32 
    
    """
    n,d=X.shape
    X_c=X-vec_mean_32(X)
    Y_c=Y-vec_mean_32(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float32)
    diag[d-1,d-1]=np.linalg.det(R.T)
    rotation=U.dot(diag).dot(VT)
    scaling=np.sum(np.abs(S.T))/np.trace(Y_c.T.dot(Y_c))
    return rotation,scaling





@nb.njit(['Tuple((float32[:,:],float32[:]))(float32[:,:],float32[:,:])'],fastmath=True,cache=True)
def recover_rotation_du_32(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot32, target
    Y: numpy array, shape (n,d), flaot32, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float32 
    scaling: numpy array, shape (d,) float32
    
    """
    n,d=X.shape
    X_c=X-vec_mean_32(X)
    Y_c=Y-vec_mean_32(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float32)
    diag[d-1,d-1]=np.linalg.det(R)
    rotation=U.dot(diag).dot(VT)
    E_list=np.eye(d,dtype=np.float32)
    scaling=np.zeros(d,dtype=np.float32)
    for i in range(d):
        Ei=np.diag(E_list[i])
        num=0
        denum=0
        for j in range(d):
            num+=X_c[j].T.dot(rotation.T).dot(Ei).dot(Y_c[j])
            denum+=Y_c[j].T.dot(Ei).dot(Y_c[j])
        scaling[i]=num/denum
    return rotation,scaling




@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64,int64)'],cache=True)
def sopt_main_32(S,T,n_iterations,N0):
    n,d=T.shape
    N1=S.shape[0]
    
    # initlize 
    rotation=np.eye(d,dtype=np.float32)    
    scalar=np.float32(1) 
    beta=vec_mean_32(T)-vec_mean_32(scalar*S.dot(rotation)) 
    #paramlist=[]
    projections=random_projections_32(d,n_iterations,1)
    mass_diff=0
    b=np.float32(np.log((N1-N0+1)/1))
    Lambda=3*np.sum(beta**2)
    rotation_list=np.zeros((n_iterations,d,d),dtype=np.float32)
    scalar_list=np.zeros((n_iterations),dtype=np.float32)
    beta_list=np.zeros((n_iterations,d),dtype=np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    Domain_org=arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/100
    for i in range(n_iterations):
#        print('i',i)
        theta=projections[i]
        T_hat_theta=np.dot(theta,T_hat.T)
        T_theta=np.dot(theta,T.T)
        
        T_hat_indice=T_hat_theta.argsort()
        T_indice=T_theta.argsort()
        T_hat_s=T_hat_theta[T_hat_indice]
        T_s=T_theta[T_indice]
        c=cost_matrix(T_hat_s,T_s)
        obj,phi,psi,piRow,piCol=solve_opt_32(c,Lambda)
        L=piRow.copy()
        L=recover_indice(T_hat_indice,T_indice,L)
        
#       debug 
#        if L.max()>=n:
#            print('error')
#            return T_hat_theta,T_theta,Lambda
#            break
        
        #move T_hat
        Domain=Domain_org[L>=0]
        mass=Domain.shape[0]
        if Domain.shape[0]>=1:
            Range=L[L>=0]
            T_hat_take_theta=T_hat_theta[Domain]
            T_take_theta=T_theta[Range]
            T_hat[Domain]+=np.expand_dims(T_take_theta-T_hat_take_theta,1)*theta

        T_hat_take=T_hat[Domain]
        S_take=S[Domain]
        
        # compute the optimal rotation, scaling, shift
        rotation,scalar=recover_rotation_32(T_hat_take,S_take)
        #scalar=np.sqrt(np.trace(np.cov(T_hat_take.T))/np.trace(np.cov(S_take.T)))
        beta=vec_mean_32(T_hat_take)-vec_mean_32(scalar*S_take.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
       
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta
        N=(N1-N0)*1/(1+b*(i/n_iterations))+N0
        mass_diff=mass-N
        if mass_diff>N*0.009:
            Lambda-=Delta 
        if mass_diff<-N*0.003:
            Lambda+=Delta
            Delta=Lambda*1/8
        if Lambda<lower_bound:
            Lambda=lower_bound
    return rotation_list,scalar_list,beta_list   




@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64,int64)'],cache=True)
def spot_bonneel_32(S,T,n_projections=20,n_iterations=200):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data
        
    n_projections: int64
        number of projections in each iteration 
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float32
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float32
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float32 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=T.shape
    N1=S.shape[0]
    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=np.float32(1) 
    beta=vec_mean_32(T)-vec_mean_32(scalar*S.dot(rotation))
    #paramlist=[]
    
    rotation_list=np.zeros((n_iterations,d,d),dtype=np.float32)
    scalar_list=np.zeros((n_iterations),dtype=np.float32)
    beta_list=np.zeros((n_iterations,d),dtype=np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
#        print('i',i)
        projections=random_projections_32(d,n_projections,1)
        T_hat=X_correspondence_pot_32(T_hat,T,projections)
        rotation,scalar=recover_rotation_32(T_hat,S)
        beta=vec_mean_32(T_hat)-vec_mean_32(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta

        #move That         
        rotation_list[i]=rotation         
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list    


@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64)'],cache=True)
def icp_du_32(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float32
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float32
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float32 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=T.shape

    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1) #
    beta=vec_mean_32(T)-vec_mean_32(scalar*np.dot(S,rotation))
    
    
    rotation_list=np.zeros((n_iterations,d,d),dtype=np.float32)
    scalar_list=np.zeros(n_iterations,dtype=np.float32)
    beta_list=np.zeros((n_iterations,d), dtype=np.float32)
    T_hat=np.dot(S,rotation)*scalar+beta
    
    # #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
#        print('i',i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M) #M.argmin(1) #closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar_d=recover_rotation_du_32(T_hat,S)
        scalar=np.mean(scalar_d)
        beta=vec_mean_32(T_hat)-vec_mean_32(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  



@nb.njit(['Tuple((float32[:,:,:],float32[:],float32[:,:]))(float32[:,:],float32[:,:],int64)'],cache=True)
def icp_umeyama_32(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float32
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float32
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float32 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=S.shape

    # initlize 
    rotation=np.eye(d,dtype=np.float32)
    scalar=nb.float32(1) 
    beta=vec_mean_32(T)-vec_mean_32(scalar*S.dot(rotation))
    # paramlist=[]
    rotation_list=np.zeros((n_iterations,d,d),dtype=np.float32)
    scalar_list=np.zeros((n_iterations),dtype=np.float32)
    beta_list=np.zeros((n_iterations,d),dtype=np.float32)
    T_hat=S.dot(rotation)*scalar+beta
        
    for i in range(n_iterations):
#        print('i',i)
       # print(i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M) #M.argmin(1) # closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar=recover_rotation_32(T_hat,S)
        #scalar=np.mean(scalar_d)
        beta=vec_mean_32(T_hat)-vec_mean_32(scalar*S.dot(rotation))
        X_hat=S.dot(rotation)*scalar+beta
        
        #move That         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  

