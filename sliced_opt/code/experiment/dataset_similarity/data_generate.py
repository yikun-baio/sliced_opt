#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:13:39 2022

@author: baly
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2 
from skimage.io import imread,imsave
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor 

import torch.optim as optim
from skimage.segmentation import slic
import os
import ot
import sys
work_path=os.path.dirname(__file__)

loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
root='experiment/dataset_similarity/data'
from sopt2.lib_set import *

#MNIST_data = datasets.MNIST(root='data',train=True,transform = ToTensor(),download=False)

MNIST_data = datasets.MNIST(
    root =root ,
    train = True,
    transform = ToTensor(), 
    download = False,
)

Fashion_data=datasets.FashionMNIST(
    root = root,
    train = True,
    transform = ToTensor(), 
    download = False,
    )

list_A=[0,1,2]
list_B=[0,7,5]
list_C=[0,1,2]
N=100
A=[]
for i in list_A:
    sample=data_extract(MNIST_data,i,N)
    A.append(sample)
A=torch.cat(A)
torch.save(A,root+'/set_A.pt')

B=[]
for i in list_B:
    sample=data_extract(MNIST_data,i,N)
    B.append(sample)
B=torch.cat(B)
torch.save(B,root+'/set_B.pt')

C=[]
for i in list_C[0:2]:
    print(i)
    sample=data_extract(MNIST_data,i,N)
    C.append(sample)
i=8
sample=data_extract(Fashion_data,i,N)
C.append(sample)
C=torch.cat(C)
torch.save(C,root+'/set_C.pt')




        
        
