#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:11:10 2022

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
num='2'
data=torch.load(root+'/data'+num+'.pt')
A=data['A']
B=data['B']
C=data['C']


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