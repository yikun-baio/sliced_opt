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



def process_data(data, X1_data, Y1_data) :
    X = data[X1_data]
    X = np.asarray(X)
    
    Y = data[Y1_data]
    Y = np.asarray(Y)
    
    return X, Y

def convert_ply(X1_data, Y1_data):
    color_x = np.array([0/255, 0/255, 255/255], dtype=np.float) # Blue
    color_y = np.array([255/255, 0/255, 0/255], dtype = np.float) # Red
    
    X = o3d.geometry.PointCloud()
    X.points = o3d.utility.Vector3dVector(X1_data)
    X.paint_uniform_color(color_x)
    
    Y = o3d.geometry.PointCloud()
    Y.points = o3d.utility.Vector3dVector(Y1_data)
    Y.paint_uniform_color(color_y)
    return X, Y

item='dragon'
data=torch.load(item+'.pt')
X1=data['X1-7p']
Y1=data['Y10-7p']


X=X1.numpy()
Y=Y1.numpy()

ply1, ply2 = convert_ply(X,Y)
dataset="lol"
ext = ".png"
vis = o3d.visualization.Visualizer()
vis.create_window(width = 1080, height= 1080, top = 100, left=50)  
ply1x = ply1.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0)) # modify the angle
ply2x = ply2.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0))# modify the angle
ply1.rotate(ply1x, center=(0,0,0))
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
