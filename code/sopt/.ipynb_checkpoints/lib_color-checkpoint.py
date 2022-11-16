
import numpy as np
import math
import torch
import os

import numba as nb 
#from numba.types import Tuple
from typing import Tuple
import sys
from numba.typed import List
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)

@nb.njit([nb.float64[:,:](nb.float64[:,:,:]),nb.float32[:,:](nb.float32[:,:,:])])
def im2mat(img):
    """Converts an image (n*m*3 matrix) to a (n*m)*3 matrix (one pixel per line)"""
    n=img.shape[0]
    m=img.shape[1]
    d=img.shape[2]
    img_a=np.ascontiguousarray(img)
    X=img_a.reshape(n*m,d)
    return X


@nb.njit(['float64[:,:,:](float64[:,:],int64,int64,int64)','float32[:,:,:](float32[:,:],int64,int64,int64)'])
def mat2im(X, n,m,d):
    """Converts back a matrix to an image"""
    X_a=np.ascontiguousarray(X)
    img=X_a.reshape(n,m,d)
    return img


@nb.njit(['float64[:,:,:](float64[:,:,:])','float32[:,:,:](float32[:,:,:])'])
def minmax(img):
    return np.clip(img, 0, 1)

