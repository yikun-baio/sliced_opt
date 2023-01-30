#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:58:45 2022

@author: baly
"""

import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import sys
import  time


# choose the data 
item='stanford_bunny'
#'/mumble_sitting' 
#'/witchcastle' #'/mumble_sitting' #'dragon' 
#'stanford_bunny'
#'dragon'
#'mumble_sitting'
#'witchcastle'
#


work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
#from sopt2.lib_shape import *

data_path=parent_path+'/experiment/shape_registration/data/test2/saved/'
save_root=data_path
data=torch.load(data_path+item+'.pt')
dtype=torch.float32
# data2=data.copy()
# for key in list(data):

#     if 'X' in key:
#         new_key=key.replace('X','T')
#         data2[new_key]=data[key]
#         del data2[key]    
#     elif 'Y' in key:
#         new_key=key.replace('Y','S')
#         data2[new_key]=data[key]
#         del data2[key]    
        
#torch.save(data2,data_path+item+'.pt')



# label='0'
# Y0=data['Y0'+label]
# X0=data['X0']

# rotation_op=data['param']['rotation_op']
# scalar_op=data['param']['scalar']
# beta_op=data['param']['beta_op']
# # add noise
# per=0.7/9
# per_s='-7p'
# Nc_y=Y0.shape[0] # of clean data
# Nc_x=X0.shape[0] # of clean data
# Nn_y=int(per*Nc_y) # of noise
# Nn_x=int(per*Nc_x) # of noise
# torch.manual_seed(0)
# nx=200*(torch.rand(Nn_x,3)-0.5)#+torch.mean(X0,0)


# time.sleep(3)
# ny=200*(torch.rand(Nn_y,3)-0.5)#+torch.mean(Y0,0)

# Y1=torch.cat((Y0,ny))
# randindex=torch.randperm(Nc_y+Nn_y)
# Y1=Y1[randindex]
# X1=torch.cat((X0,nx))

# fig = plt.figure(figsize=(10,10))
# ncolors = len(plt.rcParams['axes.prop_cycle'])
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
# ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
# plt.axis('off')
# ax.set_facecolor("grey")
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# plt.legend(loc='upper right',scatterpoints=100)

# ax.view_init(10,5,'y')

# plt.show()
# plt.close()

# data['X1'+per_s]=X1
# data['Y1'+label+per_s]=Y1 
