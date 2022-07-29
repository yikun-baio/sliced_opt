#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:58:45 2022

@author: baly
"""

import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import sys
import  time


# choose the data 
item='/stanford_bunny'
#'/witchcastle' #'/mumble_sitting' #'dragon' 
#'stanford_bunny'
#'dragon'
#'mumble_sitting'
#'witchcastle'
#'mumble_sitting' 


work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.lib_shape import *

data_path=parent_path+'/experiment/shape_registration/data/test2/saved/'
save_root=data_path
data=torch.load(data_path+item+'.pt')
dtype=torch.float32

label='1'
Y0=data['Y0'+label]
X0=data['X0']

# add noise
per=0.5/9
per_s='-5p'
Nc_y=Y0.shape[0] # of clean data
Nc_x=X0.shape[0] # of clean data
Nn_y=int(per*Nc_y) # of noise
Nn_x=int(per*Nc_x) # of noise
torch.manual_seed(0)
nx=0.8*(torch.rand(Nn_x,3)-0.5)#+torch.mean(X0,0)


time.sleep(3)
ny=0.8*(torch.rand(Nn_y,3)-0.5)#+torch.mean(Y0,0)

Y1=torch.cat((Y0,ny))
randindex=torch.randperm(Nc_y+Nn_y)
Y1=Y1[randindex]
X1=torch.cat((X0,nx))

fig = plt.figure(figsize=(10,10))
ncolors = len(plt.rcParams['axes.prop_cycle'])
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
plt.axis('off')
ax.set_facecolor("grey")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.legend(loc='upper right',scatterpoints=100)
# ax.set_xlim3d(-45,45)
# ax.set_ylim3d(-30,30)
# ax.set_zlim3d(0,60)
ax.view_init(10,5,'y')
# ax.set_xlim3d(-0.08,0.12)
# ax.set_ylim3d(0.06,0.2)
# ax.set_zlim3d(-0.02,0.14)
# ax.view_init(15,15,'y')
#ax.view_init(-10,90,'z')
plt.show()
plt.close()

#vis = o3d.visualization.Visualizer()
#fov_step=31

# pcd_X1 = o3d.geometry.PointCloud()
# pcd_X1.points = o3d.utility.Vector3dVector(X1.numpy())
# pcd_X1.paint_uniform_color([0.1, 0.1, 1])
# pcd_Y1 = o3d.geometry.PointCloud()
# pcd_Y1.points = o3d.utility.Vector3dVector(Y1.numpy())
# pcd_Y1.paint_uniform_color([0.8, 0.1, 0.1])

# vis.create_window(visible=False) #works for me with False, on some systems needs to be true
# vis.add_geometry(pcd_X1)
# vis.add_geometry(pcd_Y1)
# vis.reset_view_point(100)
# vis.get_render_option().background_color=np.array([0.6, 0.6, 0.6])

# ctr = vis.get_view_control()
# ctr.rotate(0,-400,0)

# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image(save_root+'/test_N.png')
# vis.destroy_window()
#nx=nx-torch.mean(nx,0)
# #ny=nx-torch.mean(ny,0)

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Y1[:,0],Y1[:,1],Y1[:,2],s=3,label='source') # plot the point (2,3,4) on the figure
# ax.scatter(X1[:,0],X1[:,1],X1[:,2],s=3,label='target') # plot the point 
# ax.set_xlim3d(-30,30)
# ax.set_ylim3d(-15,15)
# ax.set_zlim3d(0,35)
# plt.legend(loc='upper right')
# plt.show()
data['X1'+per_s]=X1
data['Y1'+label+per_s]=Y1 


# data['param']=param

#rotation_es,scalar_es=recover_rotation(X1,Y1)
#scalar_es=torch.sqrt(torch.trace(torch.cov(X0.T))/torch.trace(torch.cov(Y0.T)))
#beta_es=torch.mean(X0,0)-torch.mean(scalar_es*Y0@rotation_es,0)
#beta_es=torch.mean(Y0)
torch.save(data,data_path+'/'+item+'.pt')
