#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:27:06 2022

@author: baly
"""


import open3d as o3d
import os
import numpy as np
import torch

def get_noise(Y0,Y1):
    N=Y1.shape[0]
    data_indices=[]
    noise_indices=[]
    for j in range(N):
        yj=Y1[j]
        if yj in Y0:
            data_indices.append(j)
        else:
            noise_indices.append(j)
    return np.array(data_indices),np.array(Noise_indices)

def process_data(data, X1_data, Y1_data) :
    X = data[X1_data]
    X = np.asarray(X)
    
    Y = data[Y1_data]
    Y = np.asarray(Y)
    return X, Y

def convert_ply(X_data,X_noise, Y_data,Y_noise):
    color_x = np.array([0/255, 0/255, 255/255], dtype=np.float) # Blue
    color_y = np.array([255/255, 0/255, 0/255], dtype = np.float) # Red
    
    ply_X= o3d.geometry.PointCloud()
    ply_X.points = o3d.utility.Vector3dVector(X_data)
    ply_X.paint_uniform_color(color_x)
    
    ply_Xn= o3d.geometry.PointCloud()
    ply_Xn.points = o3d.utility.Vector3dVector(X_data)
    ply_Xn.paint_uniform_color(color_x)
    
    ply_Y = o3d.geometry.PointCloud()
    ply_Y.points = o3d.utility.Vector3dVector(Y_data)
    Y.paint_uniform_color(color_y)
    
    ply_Yn = o3d.geometry.PointCloud()
    ply_Yn.points = o3d.utility.Vector3dVector(Y_noise)
    Y.paint_uniform_color(color_y)
    return ply_X, ply_Xn, ply_Y, ply_Yn

item='dragon'
data=torch.load(item+'.pt')
X0=data['X0']
Y0=data['Y00']
X1=data['X1-7p']
Y1=data['Y10-7p']
n=X1.shape[0]

data_indices_X=range(0,10000)
noise_indices_X=range(10000,n)

data_indices_Y,noise_indices_Y=get_noise(Y0,Y1)

X_data=X1.numpy()[data_indices_X]
X_noise=X1.numpy()[noise_indices_X]

Y_data=Y1.numpy()[data_indices_Y]
Y_noise=Y1.numpy()[noise_indices_Y]

ply_X,ply_Xn,ply_Y,ply_Yn = convert_ply(X_data,X_noise,Y_data,Y_noise)
dataset="lol"
ext = ".png"
vis = o3d.visualization.Visualizer()
vis.create_window(width = 1080, height= 1080, top = 100, left=50)  
ply_Xr = ply_X.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0)) # modify the angle
ply_Xnr = ply_Xn.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0)) # modify the angle
ply_Yr = ply_Y.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0))# modify the angle
ply_Ynr = ply_Yn.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0))# modify the angle
ply_X.rotate(ply_Xr, center=(0,0,0))
ply_Xn.rotate(ply_Xnr, center=(0,0,0))

ply2.rotate(ply2x, center=(0,0,0))


vis.add_geometry(ply1, reset_bounding_box = True)
vis.add_geometry(ply2, reset_bounding_box = True)
#o3d.visualization.draw_geometries([ply1, ply2])

vis.get_render_option().point_size = 8   # modify the size of point 
vis.get_view_control().translate(10, 50, 0, 0) # modify the location 
vis.get_render_option().background_color = [211/255, 211/255, 211/255] # modify the background color 
name=item
o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.17)
vis.capture_screen_image(name + ext, do_render= True)
vis.destroy_window()
