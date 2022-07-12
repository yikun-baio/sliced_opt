#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:13:39 2022

@author: baly
"""


import matplotlib.pyplot as plt
import numpy as np
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
num='1'
#MNIST_data = datasets.MNIST(root='data',train=True,transform = ToTensor(),download=False)

MNIST_data = datasets.MNIST(
    root =root ,
    train = True,
    transform = ToTensor(), 
    download = True,
)

Fashion_data=datasets.FashionMNIST(
    root = root,
    train = True,
    transform = ToTensor(), 
    download = True,
    )

list_A=[0,1,2]
list_B=[0,3,7]
list_C=[0,1,3]
N=200
A=[]
for i in list_A:
    sample=data_extract(MNIST_data,i,N)
    A.append(sample)
A=torch.cat(A)


B=[]
for i in list_B:
    sample=data_extract(MNIST_data,i,N)
    B.append(sample)
B=torch.cat(B)


C=[]
for i in list_C[0:2]:
    print(i)
    sample=data_extract(MNIST_data,i,N)
    C.append(sample)
i=8
sample=data_extract(Fashion_data,i,N)
C.append(sample)
C=torch.cat(C)
data={}
data['A']=A
data['B']=B
data['C']=C

torch.save(data,root+'/data'+num+'.pt')


d = 20

encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
encoder.load_state_dict(torch.load(root+'/encoder.pt',map_location='cpu'))
decoder.load_state_dict(torch.load(root+'/decoder.pt',map_location='cpu'))

Ae=encoder.forward(A)
Be=encoder.forward(B)
Ce=encoder.forward(C)
datae={}
datae['Ae']=Ae
datae['Be']=Be
datae['Ce']=Ce
torch.save(datae,root+'/datae'+num+'.pt')



        
        
