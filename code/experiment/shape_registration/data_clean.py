#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:36:23 2022

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
work_path=os.path.dirname(__file__)
print('work_path is', work_path)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.lib_shape import *


root = parent_path+'/experiment/shape_registration/data/test2'
#save_root = os.path.join(root+'/testdata', dataset_name)
save_root=root+'/saved'
# if not os.path.exists(save_root):
#     os.makedirs(save_root)

# from experiment.shape_registration.dataset import Dataset
# dataset_name = 'modelnet40'

# choose split type from 'train', 'test', 'all', 'trainval' and 'val'
# only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
#split = 'train'

#d = Dataset(root=root, dataset_name=dataset_name, num_points=10000, split=split)
#print("datasize:", d.__len__())

    
#item='/witchcastle.txt'
#'mumble_sitting.txt'
#'dragon.ply' #'mumble_sitting' #'witchcastle'

# def load(path):
#     """takes as input the path to a .pts and returns a list of 
# 	tuples of floats containing the points in in the form:
# 	[(x_0, y_0, z_0),
# 	 (x_1, y_1, z_1),
# 	 ...
# 	 (x_n, y_n, z_n)]"""
#     with open(path) as f:
#         rows = [rows.strip() for rows in f]
#     n=len(rows)-1
#     d=3
#     L=np.zeros((n,d),dtype=np.float32)
#     for i in range(1,n+1):
#         row=rows[i].split(' ')[1:]
#         d=len(row)
#         row_d=np.zeros(d,dtype=np.float32)
#         for j in range(0,d):
#             row_d[j]=np.float32(row[j])
#         L[i-1]=row_d
#     return L


# path=root+'/'+item
# #data0=load(path)
# pcd = o3d.io.read_point_cloud(path)
# data0=np.float32(np.asarray(pcd.points))


# data0=torch.from_numpy(data0)

# n,d=data0.shape
# N=10*int(1e3)
# randint=torch.randint(0,n,(N,))
# X0=data0[randint]

# theta=torch.tensor([1/5*torch.pi,1/5*torch.pi,-1/6*torch.pi])
# rotation=rotation_3d_2(theta,'in')
# scalar=0.6 #0.6
# beta=torch.tensor([-0.02,0.02,0.01],dtype=torch.float32) #torch.tensor([1.8,0.5,0.5])
# Y0=scalar*X0@rotation+beta

# def custom_draw_geometry():
#     # The following code achieves the same effect as:
#     # o3d.visualization.draw_geometries([pcd])
#     vis = o3d.visualization.Visualizer()
#     fov_step=31
#     vis.create_window(visible=False) #works for me with False, on some systems needs to be true
#     vis.add_geometry(pcd_X0)
#     vis.add_geometry(pcd_Y0)
#     vis.reset_view_point(100)
#     vis.get_render_option().background_color=np.array([0.6, 0.6, 0.6])
    
#     ctr = vis.get_view_control()
#     ctr.rotate(0,-400,0)
    
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(save_root+'/test1.png')
#     vis.destroy_window()
    
# pcd_X0 = o3d.geometry.PointCloud()
# pcd_X0.points = o3d.utility.Vector3dVector(X0.numpy())
# pcd_X0.paint_uniform_color([0.1, 0.1, 1])
# pcd_Y0 = o3d.geometry.PointCloud()
# pcd_Y0.points = o3d.utility.Vector3dVector(Y0.numpy())
# pcd_Y0.paint_uniform_color([0.8, 0.1, 0.1])
# #custom_draw_geometry(X0)

# vis = o3d.visualization.Visualizer()
# fov_step=31
# vis.create_window(visible=False) #works for me with False, on some systems needs to be true
# vis.add_geometry(pcd_X0)
# vis.add_geometry(pcd_Y0)
# vis.reset_view_point(100)
# vis.get_render_option().background_color=np.array([0.6, 0.6, 0.6])

# ctr = vis.get_view_control()
# ctr.rotate(100,-350,100)
# ctr.scale(2)

#print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
#ctr.change_field_of_view(step=40)
#print("Field of view (after changing) %.2f" % ctr.get_field_of_view())

# vis.poll_events()
# vis.update_renderer()
# vis.capture_screen_image(save_root+'/test.jpg')
# vis.destroy_window()
#vis.run()

#vis.destroy_window()
    

# #o3d.visualization.draw_geometries([pcd_X0,pcd_Y0])
# #o3d.io.write_point_cloud(save_root+'/'+item+'.png', pcd_X0)
# fig = plt.figure(figsize=(10,10))
# ncolors = len(plt.rcParams['axes.prop_cycle'])
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X0[:,0],X0[:,1],X0[:,2],s=2,label='target',color='blue') # plot the point (2,3,4) on the figure
# ax.scatter(Y0[:,0],Y0[:,1],Y0[:,2],s=2,label='source',color='red') # plot the point (2,3,4) on the figure
# plt.axis('off')
# ax.set_facecolor("grey")
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.view_init(10,5,'y')
# plt.legend(loc='upper right',scatterpoints=100)
# # ax.set_xlim3d(-45,45)
# # ax.set_ylim3d(-30,30)
# # ax.set_zlim3d(0,60)
# plt.show()
# plt.close()

#dragon
# ax.set_xlim3d(-0.08,0.12)
# ax.set_ylim3d(0.06,0.2)
# ax.set_zlim3d(-0.02,0.14)

#mubble_sitting
# ax.set_xlim3d(-45,45)
# ax.set_ylim3d(-30,30)
# ax.set_zlim3d(0,60)

#castle
#ax.set_xlim3d(-25,25)
#ax.set_ylim3d(-15,15)
#ax.set_zlim3d(0,24)




# data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
# save_root=data_path
# data=torch.load(data_path+item+'.pt')
# dtype=torch.float32

# label='0'

# Y0=data['Y0'+label]
# X0=data['X0']
# #N=Y0.shape[0]
# N=10000
# N1=9*int(1e3)
# N2=8*int(1e3)
# #N3=9*int(1e3)

# randint=torch.randint(0,N,(N1,))
# Y01=Y0[randint]
# randint=torch.randint(0,N,(N2,))
# Y02=Y0[randint]
# #randint=torch.randint(0,N,(N3,))
# #Y03=Y0[randint]

# # #N4=10*int(1e3)
# # theta_op=-theta
# # rotation_op=rotation_3d_2(theta_op,'re')
# # scalar_op=1/scalar
# # beta_op=-1/scalar*beta@rotation_op
# # #randint=torch.randint(0,n,(N1,))
# # X0=Y0@rotation_op*scalar_op+beta_op
# # data={}
# data['X0']=X0
# data['Y00']=Y0
# data['Y01']=Y01
# data['Y02']=Y02
# #data['Y03']=Y03
# param={}

# param['theta']=theta
# param['beta']=beta
# param['scalar']=scalar
# param['rotation_op']=rotation_op
# param['scalar_op']=scalar_op
# param['beta_op']=beta_op

#data['param']=param



# # test if the data is symmetric 
# #recover_rotation(X,Y)

#torch.save(data,save_root+'/'+str(item)+'.pt')

# #X1T_c=X-torch.mean(X,0)
# #Y1T_c=Y-torch.mean(Y,0)
# #U1,S1,V1=torch.pca_lowrank(X1T_c)
# #U2,S2,V2=torch.pca_lowrank(Y1T_c)



    
    
    
    
