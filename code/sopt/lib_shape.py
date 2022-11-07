#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:25:29 2022

@author: baly
"""
import torch 
import os
import sys
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


from sopt.library import *



def get_swiss(N=100,a = 4,r_min = 0.1,r_max = 1):  
  theta = np.linspace(0, a * np.pi, N)
  r = np.linspace(r_min, r_max, N)
  X = np.stack([r * np.cos(theta),r * np.sin(theta)],1)
  return X


def rotation_matrix(theta):
      return torch.stack([torch.cos(theta).reshape([1]),torch.sin(theta).reshape([1]),
            -torch.sin(theta).reshape([1]),torch.cos(theta).reshape([1])]).reshape([2,2])

def scalar_matrix(scalar):
  return torch.stack([scalar[0:2],scalar[1:3]])




def rotation_matrix_3d_x(theta_x):
    device=theta_x.device.type
    rotation_x=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_x[1,1]=torch.cos(theta_x)
    rotation_x[1,2]=-torch.sin(theta_x)
    rotation_x[2,1]=torch.sin(theta_x)
    rotation_x[2,2]=torch.cos(theta_x)
    rotation_x[0,0]=1.0
    return rotation_x


def rotation_matrix_3d_y(theta_y):
    device=theta_y.device.type
    rotation_y=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_y[0,0]=torch.cos(theta_y)
    rotation_y[0,2]=torch.sin(theta_y)
    rotation_y[2,0]=-torch.sin(theta_y)
    rotation_y[2,2]=torch.cos(theta_y)
    rotation_y[1,1]=1.0
    return rotation_y

def rotation_matrix_3d_z(theta_z):
    device=theta_z.device.type
    rotation_z=torch.zeros((3,3),dtype=torch.float64,device=device)
    rotation_z[0,0]=torch.cos(theta_z)
    rotation_z[0,1]=-torch.sin(theta_z)
    rotation_z[1,0]=torch.sin(theta_z)
    rotation_z[1,1]=torch.cos(theta_z)
    rotation_z[2,2]=1.0
    return rotation_z

def rotation_matrix_3d(theta,order='re'):
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
    cos_x,cos_y,cos_z=torch.cos(theta)
    sin_x,sin_y,sin_z=torch.sin(theta)

    if order=='re':
        M=rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    elif order=='in':
        M=rotation_in(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z)
    return M

def rotation_re(cos_x,sin_x,cos_y,sin_y,cos_z,sin_z):
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

def recover_angle(M,order='re'):
    if order=='re':
        sin_y=M[0,2]
        cos_y=torch.sqrt(1-sin_y**2)
        sin_z=-M[0,1]/cos_y  
        cos_z=M[0,0]/cos_y  
        sin_x=-M[1,2]/cos_y
        cos_x=M[2,2]/cos_y
        theta_x=cosine_to_angle(sin_x,cos_x)
        theta_y=cosine_to_angle(sin_y,cos_y)
        theta_z=cosine_to_angle(sin_z,cos_z)
    return torch.tensor([theta_x,theta_y,theta_z])
    

def cosine_to_angle(sin_x,cos_x):
    x1=torch.arcsin(abs(sin_x))
    x2=torch.arccos(abs(cos_x))
   
    if abs(x1-x2)>=1e-4:
        print('error')
        return 'error'
    else:
        x=x1
        
    if sin_x>=0 and cos_x>=0:
        return x
    elif sin_x>=0 and cos_x<0:
        return torch.pi-x
    elif sin_x<0 and cos_x>=0:
        return -x
    elif sin_x<0 and cos_x<0:
        return -(torch.pi-x)
        


    

def order_vector(v,X):
    Xv=X@v  
    mid=(Xv.max()+Xv.min())/2  
    N1=torch.sum(Xv>mid)  
    N2=torch.sum(Xv<mid)  
    if N1<N2*1.01:    
        return 1  
    elif N1>N2*1.01:    
        return -1
    elif N1==N2:
        print('error, the data is symmetric')
        return 0

    
    
def recover_rotation(X,Y):
    n,d=X.shape
    X_c=X-torch.mean(X,0)
    Y_c=Y-torch.mean(Y,0)
    YX=Y_c.T@X_c
    U,S,VT=torch.linalg.svd(YX)
    R=U@VT
    diag=torch.eye(d)
    diag[d-1,d-1]=torch.det(R.T)
    rotation=U@diag@VT
    scaling=torch.sum(torch.abs(S.T))/torch.trace(Y_c.T@Y_c)
    return rotation,scaling

@nb.njit(['float64[:](float64[:,:])'],parallel=True,fastmath=True)
def vec_mean(X):
    n,d=X.shape
    mean=np.zeros(d,dtype=np.float64)
    for i in nb.prange(d):
        mean[i]=X[:,i].mean()
    return mean
        

@nb.njit(['Tuple((float64[:,:],float64))(float64[:,:],float64[:,:])'])
def recover_rotation_nb(X,Y):
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

@nb.njit(['Tuple((float64[:,:],float64[:]))(float64[:,:],float64[:,:])'],fastmath=True)
def recover_rotation_du_nb(X,Y):
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



def recover_rotation_du(X,Y):
    n,d=X.shape
    X_c=X-torch.mean(X,0)
    Y_c=Y-torch.mean(Y,0)
    YX=Y_c.T@X_c
    U,S,VT=torch.linalg.svd(YX)
    R=U@VT
    diag=torch.eye(d)
    diag[d-1,d-1]=torch.det(R)
    rotation=U@diag@VT
    E_list=torch.eye(3)
    scaling=torch.zeros(3)
    for i in range(3):
        Ei=torch.diag(E_list[i])
        num=0
        denum=0
        for j in range(3):
            num+=X_c[j].T@rotation.T@Ei@Y_c[j]
            denum+=Y_c[j].T@Ei@Y_c[j]
        scaling[i]=num/denum

    return rotation,scaling

def int_rotation(X,Y):
    n,d=X.shape
    X_c=X-torch.mean(X,0)
    Y_c=Y-torch.mean(Y,0)
    Ux,Sx,VTx=torch.linalg.svd(X_c)
    Uy,Sy,VTy=torch.linalg.svd(Y_c)
    R=VTy.T@VTx
    return R
    

def init_angle(X,Y):
    R_es=recover_rotation(X,Y)
    theta_es=recover_angle(R_es)
    return theta_es


    
    

