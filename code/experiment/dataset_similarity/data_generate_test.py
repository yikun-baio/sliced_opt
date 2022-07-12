#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:19:50 2022

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
num='test'
#MNIST_data = datasets.MNIST(root='data',train=True,transform = ToTensor(),download=False)

MNIST_data = datasets.MNIST(
    root =root,
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


N=200
A=[]
for i in range(10):
    sample=data_extract(MNIST_data,i,N)
    A.append(sample)
    
for i in range(10):
    sample=data_extract(Fashion_data,i,N)
    A.append(sample)
A=torch.cat(A)



B=[]
for i in range(10):
    sample=data_extract(MNIST_data,i,N)
    B.append(sample)
for i in range(10):
    sample=data_extract(Fashion_data,i,N)
    B.append(sample)
B=torch.cat(B)
data={}
data['A']=A
data['B']=B

torch.save(data,root+'/data'+num+'.pt')


d = 32
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
encoder.load_state_dict(torch.load(root+'/encoder.pt',map_location='cpu'))
decoder.load_state_dict(torch.load(root+'/decoder.pt',map_location='cpu'))

Ae=encoder.forward(A)
Be=encoder.forward(B)

datae={}
datae['Ae']=Ae
datae['Be']=Be

torch.save(datae,root+'/datae'+num+'.pt')