#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:24:08 2022

"""

import sys
import open3d as o3d
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import ot

work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
item='/stanford_bunny'

exp_num=item
from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *
label='1'
n_point='/9k'
per_s='-5p'
data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
save_path='experiment/shape_registration/result'+exp_num+n_point+per_s
method='/sopt'
result=torch.load(save_path+method+'_param.pt')
data=torch.load(data_path+item+'.pt')

X0=data['X0'].to(torch.float32)
Y0=data['Y0'+label].to(torch.float32)
X1=data['X1'+per_s].to(torch.float32)
Y1=data['Y1'+label+per_s].to(torch.float32)
print('Y1.shape is', Y1.shape)

n_iteration=len(param)