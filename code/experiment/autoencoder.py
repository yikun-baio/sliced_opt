#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:59:19 2022

@author: baly
"""

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem
from tqdm import tqdm # this module is useful to plot progress bars
import plotly.io as pio
pio.renderers.default = 'colab'

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.lib_set import *
root='experiment/dataset_similarity'
device='cpu'
num='1'
data=torch.load(root+'/data/data'+num+'.pt')
A=data['A']
B=data['B']
C=data['C']
n=20
N=n*200
#A=A.reshape([N,1,28,28])
#B=B.reshape([N,1,28,28])
#C=C.reshape([N,1,28,28])
d = 32

encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
encoder.load_state_dict(torch.load(root+'/encoder.pt', map_location=device))
decoder.load_state_dict(torch.load(root+'/decoder.pt', map_location=device))
Ae=encoder.forward(A).detach()
Be=encoder.forward(B).detach()
Ce=encoder.forward(C).detach()
datae={}
datae['Ae']=Ae
datae['Be']=Be
datae['Ce']=Ce
torch.save(datae,root+'/data/data'+num+'e.pt')


