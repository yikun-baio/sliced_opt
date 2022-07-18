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
num='5'
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

# list_A=[3,5,8]
# list_B=[3,0,2]
# list_C=[3,5,3]

list_A=[0,3,5]
list_B=[0,2,8]
list_C=[0,3,1]
N=200
A=[]

for i in list_A:
    sample=data_extract(MNIST_data,i,N)
    A.append(sample)
A=torch.cat(A)
#A=A.reshape((N*len(list_A),1,28,28))

C=[]
for i in list_C[0:2]:
    sample=data_extract(MNIST_data,i,N)
    C.append(sample)
i=list_C[-1]
sample=data_extract(Fashion_data,i,N)
C.append(sample)
C=torch.cat(C)

B=[]
for i in list_B:
    torch.manual_seed(i)
    sample=data_extract(MNIST_data,i,N)
    B.append(sample)
B=torch.cat(B)



#i=list_C[2]
#

data={}
data['A']=A
data['B']=B
data['C']=C



Lambda='_1'
d = 32

encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
encoder.load_state_dict(torch.load(root+'/encoder' +Lambda+'.pt',map_location='cpu'))
decoder.load_state_dict(torch.load(root+'/decoder'+Lambda+'.pt',map_location='cpu'))

encoder.eval()
Ae=encoder(A)
Be=encoder(B)
Ce=encoder(C)

data['Ae']=Ae
data['Be']=Be
data['Ce']=Ce
torch.save(data,root+'/test/data'+num+'.pt')



        
        
