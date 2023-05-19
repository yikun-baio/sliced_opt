#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022
@author: Yikun Bai yikun.bai@Vanderbilt.edu 
"""
import torch 
import os
import sys


from .library import *
from .lib_ot import *   
from .sliced_opt import * 
import matplotlib.pyplot as plt

# def get_swiss(N=100,a = 4,r_min = 0.1,r_max = 1): 
#     """
#     generate swiss shape data
#     parameters: 
#     -------
#     N : int64 
#     a: float or int64
#     r_min: float
#     r_max: float

#     returns:
#     ------
#     X: numpy array, shape (N,2), float64 
#     """

#   theta = np.linspace(0, a * np.pi, N)
#   r = np.linspace(r_min, r_max, N)
#   X = np.stack([r * np.cos(theta),r * np.sin(theta)],1)
#   return X


def rotation_matrix(theta):
    """
    generate (2,2) rotation matrix
    
    Parameter:
    ------
    theta : float
    
    Returns: 
    -------
    torch.tensor shape (2,2) float 
    """
    return torch.stack([torch.cos(theta).reshape([1]),torch.sin(theta).reshape([1]),
            -torch.sin(theta).reshape([1]),torch.cos(theta).reshape([1])]).reshape([2,2])





def rotation_matrix_3d_x(theta_x):
    """
    generate (3,3) rotation matrix along x-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
    device=theta_x.device.type
    rotation_x=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_x[1,1]=torch.cos(theta_x)
    rotation_x[1,2]=-torch.sin(theta_x)
    rotation_x[2,1]=torch.sin(theta_x)
    rotation_x[2,2]=torch.cos(theta_x)
    rotation_x[0,0]=1.0
    return rotation_x


def rotation_matrix_3d_y(theta_y):
    """
    generate (3,3) rotation matrix along y-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    torch.tensor shape (3,3) float 
    """
        
    device=theta_y.device.type
    rotation_y=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_y[0,0]=torch.cos(theta_y)
    rotation_y[0,2]=torch.sin(theta_y)
    rotation_y[2,0]=-torch.sin(theta_y)
    rotation_y[2,2]=torch.cos(theta_y)
    rotation_y[1,1]=1.0
    return rotation_y

def rotation_matrix_3d_z(theta_z):
    """
    generate (3,3) rotation matrix along z-axis 
    
    Parameter:
    -----
    theta: float
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    device=theta_z.device.type
    rotation_z=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_z[0,0]=torch.cos(theta_z)
    rotation_z[0,1]=-torch.sin(theta_z)
    rotation_z[1,0]=torch.sin(theta_z)
    rotation_z[1,1]=torch.cos(theta_z)
    rotation_z[2,2]=1.0
    return rotation_z

def rotation_matrix_3d(theta,order='re'):
    
    """
    generate (3,3) rotation matrix 
    
    Parameter:
    -----
    theta: torch tensor (3,) float
    order: string "re" or "in" 
          "in" roation with respect to x-axis, then y-axis, then z-axis
          "re" rotation with rspect to z-axis, then y-axis, then x-axis 
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    theta_x,theta_y,theta_z=theta
    rotatioin_x=rotation_matrix_3d_x(theta_x)
    rotatioin_y=rotation_matrix_3d_y(theta_y)
    rotatioin_z=rotation_matrix_3d_z(theta_z)
    if order=='in':
        rotation_3d=torch.linalg.multi_dot((rotatioin_z,rotatioin_y,rotatioin_x))
    elif order=='re':
        rotation_3d=torch.linalg.multi_dot((rotatioin_x,rotatioin_y,rotatioin_z))
    return rotation_3d

def rotation_3d_2(theta,order='re'):
    """
    generate (3,3) rotation matrix 
    
    Parameter:
    -----
    theta: torch tensor (3,) float
    order: string "re" or "in" 
          "in" roation with respect to x-axis, then y-axis, then z-axis
          "re" rotation with rspect to z-axis, then y-axis, then x-axis 
    Returns: 
    ------
    M: torch.tensor shape (3,3) float 
    """
        
    cos_x,cos_y,cos_z=torch.cos(theta)
    sin_x,sin_y,sin_z=torch.sin(theta)

    if order=='re':
        M=rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    elif order=='in':
        M=rotation_in(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    return M

def rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to z-axis, then y-axis, then x-axis
    """
    
    M=torch.zeros((3,3),dtype=torch.float64)
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_y*sin_z
    M[0,2]=sin_y
    M[1,0]=sin_x*sin_y*cos_z+cos_x*sin_z
    M[1,1]=-sin_x*sin_y*sin_z+cos_x*cos_z
    M[1,2]=-sin_x*cos_y
    M[2,0]=-cos_x*sin_y*cos_z+sin_x*sin_z
    M[2,1]=cos_x*sin_y*sin_z+sin_x*cos_z 
    M[2,2]=cos_x*cos_y
    return M

def rotation_in(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z):
    """
    generate (3,3) rotation matrix along  
    
    Parameter:
    -----
    cos_x,sin_x: float,float
                cos(x), sin(x) for some angle x
    cos_y,sin_y: float, float
                cos(y), sin(y) for some angle y
    cos_z,sin_z: float, float
                cos(z), sin(z) for some angle z
    
    Returns: 
    ------
    M: torch.tensor shape (3,3) float
             rotation with rspect to x-axis, then y-axis, then z-axis
    """
    M=torch.zeros((3,3))
    M[0,0]=cos_y*cos_z
    M[0,1]=-cos_x*sin_z+sin_x*sin_y*cos_z
    M[0,2]=sin_x*sin_z+cos_x*sin_y*cos_z
    M[1,0]=cos_y*sin_z
    M[1,1]=cos_x*cos_z+sin_x*sin_y*sin_z
    M[1,2]=-sin_x*cos_z+cos_x*sin_y*sin_z
    M[2,0]=-sin_y
    M[2,1]=sin_x*cos_y
    M[2,2]=cos_x*cos_y
    return M


    

    

@nb.njit(['float64[:](float64[:,:])'],fastmath=True,cache=True)
def vec_mean(X):
    """
    return X.mean(1) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64
    
    Return:
    --------
    mean: numpy array, shape (d,), float64 
    
    
    """
    n,d=X.shape
    mean=np.zeros(d,dtype=np.float64)
    for i in nb.prange(d):
        mean[i]=X[:,i].mean()
    return mean
        


    
    
@nb.njit(['Tuple((float64[:,:],float64))(float64[:,:],float64[:,:])'],cache=True)
def recover_rotation(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: float64 
    
    """

        
    n,d=X.shape
    X_c=X-vec_mean(X)
    Y_c=Y-vec_mean(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float64)
    diag[d-1,d-1]=np.linalg.det(R.T)
    rotation=U.dot(diag).dot(VT)
    scaling=np.sum(np.abs(S.T))/np.trace(Y_c.T.dot(Y_c))
    return rotation,scaling



@nb.njit(['Tuple((float64[:,:],float64[:]))(float64[:,:],float64[:,:])'],fastmath=True,cache=True)
def recover_rotation_du(X,Y):
    """
    return the optimal rotation, scaling based on the correspondence (X,Y) 
    
    Parameters:
    ----------
    X: numpy array, shape (n,d), flaot64, target
    Y: numpy array, shape (n,d), flaot64, source
    
    Return:
    --------
    rotation: numpy array, shape (d,d), float64 
    scaling: numpy array, shape (d,) float64 
    
    """
    
    n,d=X.shape
    X_c=X-vec_mean(X)
    Y_c=Y-vec_mean(Y)
    YX=Y_c.T.dot(X_c)
    U,S,VT=np.linalg.svd(YX)
    R=U.dot(VT)
    diag=np.eye(d,dtype=np.float64)
    diag[d-1,d-1]=np.linalg.det(R)
    rotation=U.dot(diag).dot(VT)
    E_list=np.eye(d,dtype=np.float64)
    scaling=np.zeros(d,dtype=np.float64)
    for i in range(d):
        Ei=np.diag(E_list[i])
        num=0
        denum=0
        for j in range(d):
            num+=X_c[j].T.dot(rotation.T).dot(Ei).dot(Y_c[j])
            denum+=Y_c[j].T.dot(Ei).dot(Y_c[j])
        scaling[i]=num/denum
    return rotation,scaling





# our method
#@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64)'])
@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64)'],cache=True)
def sopt_main(S,T,n_iterations,N0):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float32
        source data 
    T: (n,d) numpy array, float32
        target data
    n_iterations: int64
        total number of iterations
    N0: int64 
        number of clean data (our prior knowledge)
    
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
    rotation=np.eye(d,dtype=np.float64)    
    scalar=np.float32(1) 
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation)) 
    #paramlist=[]
    projections=random_projections(d,n_iterations,1)
    mass_diff=0
    b=np.log((N1-N0+1)/1)
    Lambda=6*np.sum(beta**2)
    rotation_list=np.zeros((n_iterations,d,d)) #,dtype=np.float32)
    scalar_list=np.zeros((n_iterations)) #,dtype=np.float32)
    beta_list=np.zeros((n_iterations,d)) #,dtype=np.float32)
    T_hat=S.dot(rotation)*scalar+beta
    Domain_org=arange(0,N1)
    Delta=Lambda/8
    lower_bound=Lambda/10000
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
        obj,phi,psi,piRow,piCol=solve_opt(c,Lambda)
        L=piRow.copy()
        L=recover_indice(T_hat_indice,T_indice,L)
        
      #debug 
        # if L.max()>=n:
        #     print('error')
        #     return T_hat_theta,T_theta,Lambda
        #     break
        
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
        rotation,scalar=recover_rotation(T_hat_take,S_take)
        scalar=np.sqrt(np.trace(np.cov(T_hat_take.T))/np.trace(np.cov(S_take.T)))
        beta=vec_mean(T_hat_take)-vec_mean(scalar*S_take.dot(rotation))
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
        if Lambda<Delta:
            Lambda=Delta
            Delta=Delta*1/2
        if Delta<lower_bound:
            Delta=lower_bound
        # if i&50==0:
        #     print('scalar',scalar)
        #     print('lambda',Lambda)
        #     print('delta',Delta)
        #     print('N',N)
        #     print('mass_diff',mass_diff)
    return rotation_list,scalar_list,beta_list   






# method of spot_boneel 
@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64,int64)'],cache=True)
def spot_bonneel(S,T,n_projections=20,n_iterations=200):
    
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
    n_projections: int64
        number of projections in each iteration 
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    
    n,d=T.shape
    N1=S.shape[0]
    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=nb.float64(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    #paramlist=[]
    
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=S.dot(rotation)*scalar+beta
    
    #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
#        print('i',i)

        projections=random_projections(d,n_projections,1)
        
# #        print('start1')
        T_hat=X_correspondence_pot(T_hat,T,projections)
        rotation,scalar=recover_rotation(T_hat,S)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta

#         #move That         
        rotation_list[i]=rotation         
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list    



@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64)'],cache=True)
def icp_du(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=T.shape

    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=1.0  #nb.float64(1) #
    beta=vec_mean(T)-vec_mean(scalar*np.dot(S,rotation))

    
    
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=np.dot(S,rotation)*scalar+beta
    
    # #Lx_hat_org=arange(0,n)
    
    for i in range(n_iterations):
#        print('i',i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M) #M.argmin(1) 
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar_d=recover_rotation_du(T_hat,S)
        scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        T_hat=S.dot(rotation)*scalar+beta
        
        #move Xhat         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  




@nb.njit(['Tuple((float64[:,:,:],float64[:],float64[:,:]))(float64[:,:],float64[:,:],int64)'],cache=True)
def icp_umeyama(S,T,n_iterations):
    '''
    Parameters: 
    ------
    S: (n,d) numpy array, float64
        source data 
    T: (n,d) numpy array, float64
        target data
        
    
    n_iterations: int64
        total number of iterations

    
    Returns: 
    -----
    rotation_list: (n_iterations,d,d) numpy array, float64
                  list of rotation matrices in all iterations
    scalar_list: (n_iterations,) numpy array, float64
                  list of scaling parameters in all interations
    beta_list: (n_iterations,d) numpy arrayy, float64 
                  list of translation parameters in all interations 
                      
    '''
        
    n,d=S.shape

    # initlize 
    rotation=np.eye(d) #,dtype=np.float64)
    scalar=1.0 #nb.float64(1.0) #
    beta=vec_mean(T)-vec_mean(scalar*S.dot(rotation))
    # paramlist=[]
    rotation_list=np.zeros((n_iterations,d,d)) #.astype(np.float64)
    scalar_list=np.zeros((n_iterations)) #.astype(np.float64)
    beta_list=np.zeros((n_iterations,d)) #.astype(np.float64)
    T_hat=S.dot(rotation)*scalar+beta
    

    
    for i in range(n_iterations):
#        print('i',i)
       # print(i)
        M=cost_matrix_d(T_hat,T)
        argmin_T=closest_y_M(M) #M.argmin(1) #closest_y_M(M)
        T_take=T[argmin_T]
        T_hat=T_take
        rotation,scalar=recover_rotation(T_hat,S)
        #scalar=np.mean(scalar_d)
        beta=vec_mean(T_hat)-vec_mean(scalar*S.dot(rotation))
        X_hat=S.dot(rotation)*scalar+beta
        
        #move That         
        rotation_list[i]=rotation
        scalar_list[i]=scalar
        beta_list[i]=beta

    return rotation_list,scalar_list,beta_list  






# def recover_rotation_du(X,Y):
#     n,d=X.shape
#     X_c=X-torch.mean(X,0)
#     Y_c=Y-torch.mean(Y,0)
#     YX=Y_c.T@X_c
#     U,S,VT=torch.linalg.svd(YX)
#     R=U@VT
#     diag=torch.eye(d)
#     diag[d-1,d-1]=torch.det(R)
#     rotation=U@diag@VT
#     E_list=torch.eye(3)
#     scaling=torch.zeros(3)
#     for i in range(3):
#         Ei=torch.diag(E_list[i])
#         num=0
#         denum=0
#         for j in range(3):
#             num+=X_c[j].T@rotation.T@Ei@Y_c[j]
#             denum+=Y_c[j].T@Ei@Y_c[j]
#         scaling[i]=num/denum

#     return rotation,scaling

# def int_rotation(X,Y):
#     n,d=X.shape
#     X_c=X-torch.mean(X,0)
#     Y_c=Y-torch.mean(Y,0)
#     Ux,Sx,VTx=torch.linalg.svd(X_c)
#     Uy,Sy,VTy=torch.linalg.svd(Y_c)
#     R=VTy.T@VTx
#     return R
    

# def init_angle(X,Y):
#     R_es=recover_rotation(X,Y)
#     theta_es=recover_angle(R_es)
#     return theta_es


def save_parameter(rotation_list,scalar_list,beta_list,save_path):
    """
    convert parameter list and save as one file 
    
    parameter:
    ------------
    rotation_list: numpy array, shape (n_itetations, d,d), float
    scalar_list: numpy array, shape (n_itetations,), float
    beta_list: numpy array, shape (n_itetations, d), float
    save_path: string 
    """
    paramlist=[]
    N=len(rotation_list)
    for i in range(N):
        param={}
        param['rotation']=rotation_list[i]
        param['beta']=beta_list[i]
        param['scalar']=scalar_list[i]
        paramlist.append(param)
    torch.save(paramlist,save_path)
    #return paramlist

    
    

# visualization 

def get_noise_index(Y0,Y1):
    """
    get the indices of clean data and noise data of Y1 
    
    Parameters:
    ---------
    Y0: numpy array, shape (N1,d), clean data
    Y1: numpy array, shape (N1+s,d), noisy data, where s>0. Y0 \subset Y1 
    
    Returns: 
    ----------
    np.array(data_indices): numpy array, shape (N1,), int64  
    np.array(noice): numpy array, shape (s,), int64 
    """
        
    N=Y1.shape[0]
    data_indices=[]
    noise_indices=[]
    for j in range(N):
        yj=Y1[j]
        if yj in Y0:
            data_indices.append(j)
        else:
            noise_indices.append(j)
    return np.array(data_indices),np.array(noise_indices)

def shape_image(T_data,S_data,T_noise=[],S_noise=[],name=None, param=None):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    if param!=None:
        xlim,ylim,zlim,view_init,(dx,dy,dz)=param
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(view_init[0],view_init[1],vertical_axis=view_init[2])
        
    ax.scatter(T_data[:,0]+dx,T_data[:,1]+dy,T_data[:,2]+dz,alpha=.5,c='C2',s=2,marker='o')
    ax.scatter(S_data[:,0]+dx,S_data[:,1]+dy,S_data[:,2]+dz,alpha=0.5,c='C1',s=2,marker='o')
    if len(T_noise)>0:
        ax.scatter(T_noise[:,0]+dx,T_noise[:,1]+dy,T_noise[:,2]+dz,alpha=.5,c='C2',s=10,marker='o')
    if len(S_noise)>0:
        ax.scatter(S_noise[:,0]+dx,S_noise[:,1]+dy,S_noise[:,2]+dz,alpha=.5,c='C1',s=10,marker='o')
    ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.axis('off')
    

    if name!=None:
        plt.savefig(name+'.pdf',dpi=200,format='pdf',bbox_inches='tight')
    plt.show()
    plt.close()

# select data 
item_list=['stanford_bunny','dragon','mumble_sitting','witchcastle']


# pamameter for ploting point cloud 
vis_param_list={'stanford_bunny':
            ([-.1,.1],[-.1,.1],[-.1,.1],(90,-90,'z'),(0.02,-0.1,0)),
'dragon':
            ([-.1,.1],[-.1,.1],[-0.1,0.1],(90,-90,'z'),(0,-0.1,0)),
'mumble_sitting':
            ([-36,36],[-36,36],[-36,36],(-20,10,'y'),(10,-10,-10)),
'witchcastle':
            ([-20,20],[-20,20],[-20,20],(45,120,'z'),(-10,20,20)),
           }    


vis_param_list0={'stanford_bunny':
            ([-.2,.2],[-.2,.2],[-.2,.2],(90,-90,'z'),(0.02,-0.1,0)),
'dragon':
            ([-.2,.2],[-.2,.2],[-0.2,0.2],(90,-90,'z'),(0,-0.1,0)),
'mumble_sitting':
            ([-66,66],[-66,66],[-66,66],(-20,10,'y'),(10,0,-10)),
'witchcastle':
            ([-38,38],[-38,38],[-38,38],(45,120,'z'),(-10,20,20)),
           }  




def init_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    """
    make a plot for the data and noise and save the plot 
    parameters: 
    X_data: numpy array, shape (n1,d), float 
        cliean data, target data 
    X_noise: numpy array, shape (s1,d), float 
        cliean data, target data 
    Y_data: numpy array, shape (n2,d), float 
        cliean data, source data 
    Y_noise: numpy array, shape (s2,d), float 
        cliean data, source data
    image_path: string 
    name: string 
        
    """
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+5,X_data[:,1],X_data[:,2]-10,alpha=.5,c='C2',s=2,marker='o')
    ax.scatter(X_noise[:,0]+5,X_noise[:,1],X_noise[:,2]-10,alpha=.5,c='C2',s=10,marker='o')
    ax.scatter(Y_data[:,0]+5,Y_data[:,1],Y_data[:,2]-10,alpha=0.5,c='C1',s=2,marker='o')
    ax.scatter(Y_noise[:,0]+5,Y_noise[:,1],Y_noise[:,2]-10,alpha=.5,c='C1',s=10,marker='o')
    ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.axis('off')
    
    # whitch_castle 
    # x+5, z-10
    ax.set_xlim([-38,38])
    ax.set_ylim([-38,38])
    ax.set_zlim([-38,38])
    ax.view_init(45,120)
    
    
    #mumble_sitting 
    # ax.set_xlim([-66,66])
    # ax.set_ylim([-66,66])
    # ax.set_zlim([-66,66])
    # ax.axis('off')
    # ax.view_init(-20,10,'y')

    
    #dragon +bunny     
    # bunny y-0.05,
    # ax.set_xlim([-.25,.25])
    # ax.set_ylim([-.25,.25])
    # ax.set_zlim([-.25,.25])
    # ax.axis('off')
    # ax.view_init( 90, -90)
    

    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    
    

def normal_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    
    """
    make a plot for the data and noise and save the plot 
    truncated version 
    parameters: 
    X_data: numpy array, shape (n1,d), float 
        cliean data, target data 
    X_noise: numpy array, shape (s1,d), float 
        cliean data, target data 
    Y_data: numpy array, shape (n2,d), float 
        cliean data, source data 
    Y_noise: numpy array, shape (s2,d), float 
        cliean data, source data
    image_path: string 
    name: string 
        
    """
        
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+3,X_data[:,1],X_data[:,2]-15,alpha=.3,c='C2',s=5,marker='o')
    ax.scatter(X_noise[:,0]+3,X_noise[:,1],X_noise[:,2]-15,alpha=.5,c='C2',s=15,marker='o')
    ax.scatter(Y_data[:,0]+3,Y_data[:,1],Y_data[:,2]-15,alpha=.9,c='C1',s=6,marker='o')
    ax.scatter(Y_noise[:,0]+3,Y_noise[:,1],Y_noise[:,2]-15,alpha=.5,c='C1',s=15,marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    ax.set_facecolor('black') 
    ax.grid(True)
    
    # castle,   
    #x+3, z-15 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])
    ax.view_init(45,120)
#    ax.view_init(0,10,'y')


    # #mumble_sitting, bunny  
    # y-10
    # ax.set_xlim([-36,36])
    # ax.set_ylim([-36,36])
    # ax.set_zlim([-36,36])
    # ax.view_init(-20,10,'y')
    #ax.view_init( 90, -90)
     
    #dragon, bunny 
    #dragon y-0.1
    #bunny, x+0.02, y-0.1    
    # ax.set_xlim([-.1,.1])
    # ax.set_ylim([-.1,.1])
    # ax.set_zlim([-.1,.1])

    # ax.view_init( 90, -90)
    # fig.set_facecolor('black')
    
    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    