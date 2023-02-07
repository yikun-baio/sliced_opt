"""
Created on Sun Jun 26 14:25:29 2022
@author: Yikun Bai Yikun.Bai@Vanderbilt.edu
"""
import numpy as np
import math
import torch
import os
import numba as nb 
#from numba.types import Tuple
from typing import Tuple
import sys
import matplotlib.pyplot as plt

from numba.typed import List

from .library import *
from .sliced_opt import *   
from .lib_ot import *   

@nb.njit([nb.float64[:,:](nb.float64[:,:,:]),nb.float32[:,:](nb.float32[:,:,:])])
def im2mat(img):
    """Converts an image (n*m*3 matrix) to a (n*m)*3 matrix (one pixel per line)"""
    n=img.shape[0]
    m=img.shape[1]
    d=img.shape[2]
    img_a=np.ascontiguousarray(img)
    X=img_a.reshape(n*m,d)
    return X

print('hello')

@nb.njit(['float64[:,:,:](float64[:,:],int64,int64,int64)','float32[:,:,:](float32[:,:],int64,int64,int64)'])
def mat2im(X, n,m,d):
    """Converts back a matrix to an image"""
    X_a=np.ascontiguousarray(X)
    img=X_a.reshape(n,m,d)
    return img


@nb.njit(['float64[:,:,:](float64[:,:,:])','float32[:,:,:](float32[:,:,:])'])
def minmax(img):
    return np.clip(img, 0, 1)


def recover_image(transp_Xs,shape,name,save_path):
    #print(transp_Xs)
    n,m,d=shape
    I1t = minmax(mat2im(transp_Xs, n,m,d))
    plot_image(I1t,name,save_path)


def plot_image(I1t,name,save_path):
    #print(transp_Xs)
    plt.figure()
    plt.axis('off')
    plt.imshow(I1t)
    #plt.pad_inces=0.01
    plt.savefig(save_path+'/'+name+'.png',format="png",dpi=800,bbox_inches='tight',pad_inches = 0)
    plt.show()
    #f.clear()
    plt.close()

    
    

    



@nb.njit(['float64[:,:](float64[:,:],float64[:,:],float64[:,:],int64)'],fastmath=True)
def transform(Xs0,Xs,Xs1,batch_size=128):    
    
    # # perform out of sample mapping
    n,m=Xs0.shape
    indices = arange(0,Xs0.shape[0])
    batch_ind = [indices[i:i + batch_size] for i in np.arange(0, len(indices), batch_size)]
    transp_Xs = np.zeros((n,m))

    for bi in batch_ind:
        # get the nearest neighbor in the source domain
        D0 = cost_matrix_d(Xs0[bi], Xs)
        idx = np.argmin(D0, axis=1)
        # define the transported points
        transp_Xs[bi] =Xs0[bi]+Xs1[idx, :]  - Xs[idx, :]
        #print(transp_Xs)
        #transp_Xs.append(transp_Xs_)
    #transp_Xs = torch.cat(transp_Xs, axis=0)
    return transp_Xs


@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],int64)'],fastmath=True)
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


@nb.njit(['float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64)'])
def spot_transfer(Xs0,Xt0,Xs,Xt,n_projections=400):
    n,d=Xs.shape
    #np.random.seed(0)
    projections=random_projections(d,n_projections,1)
    Xsc=Xs.copy()
    X_correspondence_pot(Xs,Xt,projections)     
    batch_size=128
    transp_Xs=transform(Xs0,Xsc,Xs,batch_size)
    return transp_Xs


@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],float32[:,:],int64)'])
def spot_transfer_32(Xs0,Xt0,Xs,Xt,n_projections=400):
    n,d=Xs.shape
    #np.random.seed(0)
    projections=random_projections_32(d,n_projections,1)
    Xsc=Xs.copy()
    X_correspondence_pot_32(Xs,Xt,projections)     
    batch_size=128
    transp_Xs=transform_32(Xs0,Xsc,Xs,batch_size)
    return transp_Xs


@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:],int64)'])
def sopt_transfer_32(Xs0,Xt0,Xs,Xt,Lambda_list,n_projections=400):    
    n,d=Xs.shape
    #np.random.seed(0)
    projections=random_projections_32(d,n_projections,1)
    Xsc=Xs.copy()
    X_correspondence_32(Xs,Xt,projections,Lambda_list)     
    batch_size=128
    transp_Xs=transform_32(Xs0,Xsc,Xs,batch_size)
    return transp_Xs

@nb.njit(['float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],int64)'])
def sopt_transfer(Xs0,Xt0,Xs,Xt,Lambda_list,n_projections=400):    
    n,d=Xs.shape
    #np.random.seed(0)
    projections=random_projections(d,n_projections,1)
    Xsc=Xs.copy()
    X_correspondence(Xs,Xt,projections,Lambda_list)     
    batch_size=128
    transp_Xs=transform(Xs0,Xsc,Xs,batch_size)
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

@nb.njit(['float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64,int64)'])
def eot_transfer_32(Xs0,Xt0,Xs,Xt,reg=0.1,numItermax=1000000):
    n,d=Xs.shape
    m=Xt.shape[0]
    #plan=ot.emd()
    # get the transporation plan
    Xsc=Xs.copy()
    M=cost_matrix_d(Xs,Xt)
    mu=np.ones(n,dtype=np.float32)/n
    nu=np.ones(m,dtype=np.float32)/m
    plan=sinkhorn_knopp(mu, nu, M, reg=reg,numItermax=numItermax)

    # get the transported Xs
    cond_plan=plan/np.expand_dims(np.sum(plan,1),1)
    Xs=np.dot(cond_plan,Xt)
    
#    # # prediction between images (using out of sample prediction as in [6])
    batch_size=128
    transp_Xs = transform(Xs0,Xsc,Xs,batch_size)
    
    return transp_Xs


@nb.njit(['float32[:,:](float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32,int64)'])
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


def ot_transfer_orig(Xs0,Xt0,Xs,Xt,max_iter=1000000):
#    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
#    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    # EMDTransport
    ot_emd = ot.da.EMDTransport(max_iter=max_iter)
    
    ot_emd.fit(Xs=Xs, Xt=Xt)
    # # prediction between images (using out of sample prediction as in [6])
    transp_Xs = ot_emd.transform(Xs=Xs0)
    return transp_Xs



# EOT-based method
def eot_transfer_orig(Xs0,Xt0,Xs,Xt,reg=0.1,max_iter=1000000):
    # SinkhornTransport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=reg,max_iter=max_iter)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    transp_Xs = ot_sinkhorn.transform(Xs=Xs0)
    return transp_Xs