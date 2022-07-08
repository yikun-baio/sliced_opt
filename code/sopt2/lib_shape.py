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


from sopt2.library import *



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
    rotation_x=torch.zeros((3,3),dtype=torch.float32,device=device)
    rotation_x[1,1]=torch.cos(theta_x)
    rotation_x[1,2]=-torch.sin(theta_x)
    rotation_x[2,1]=torch.sin(theta_x)
    rotation_x[2,2]=torch.cos(theta_x)
    rotation_x[0,0]=1.0
    return rotation_x


def rotation_matrix_3d_y(theta_y):
    device=theta_y.device.type
    rotation_y=torch.zeros((3,3),dtype=torch.float32,device=device)
    rotation_y[0,0]=torch.cos(theta_y)
    rotation_y[0,2]=torch.sin(theta_y)
    rotation_y[2,0]=-torch.sin(theta_y)
    rotation_y[2,2]=torch.cos(theta_y)
    rotation_y[1,1]=1.0
    return rotation_y

def rotation_matrix_3d_z(theta_z):
    device=theta_z.device.type
    rotation_z=torch.zeros((3,3),dtype=torch.float32,device=device)
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
    M=torch.zeros((3,3))
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
    X_c=X-torch.mean(X,0)
    Y_c=Y-torch.mean(Y,0)
    YX=Y_c.T@X_c
    U,S,VT=torch.linalg.svd(YX)
    R=U@VT
    return R



def init_angle(X,Y):
    R_es=recover_rotation(X,Y)
    theta_es=recover_angle(R_es)
    return theta_es
    
    
    

