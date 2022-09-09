#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:07:46 2022

@author: baly
"""
import open3d as o3d
from result_generated import *
import argparse
import os
from helpers import *


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


models = ['sopt_param', 'spot_param', 'icp_du_param', 'icp_umeyama_param']
ver = ['9k-5p', '9k-7p', '10k-5p', '10k-7p']
datas = ['witchcastle']

data=datas[0]
model=models[0]
version=ver[0]
args={}
print (f"************  Data : {data}\t Model : {model}\t Version : {version}")
args['para_path'] = "../Parameters/" + data + "/" + version + "/"
args['model_name'] = model
data_path=args['data_path'] = "../Data/"
args['data'] = data
args['data_version'] = version
args['saved_path'] = "../Images/"+ args['data'] + "/"  + args['data_version'] + "/" + args['model_name'] + "/"
param_list = open_paras(args['para_path'], args['model_name'])
X, Y = get_type(args['data_version'])
data_path = args['data_path'] + args['data']
X1 = X
Y1 = Y 
saved_path = args['saved_path']

N = len(param_list)
data = hlp.load_data(data_path)
X, Y = hlp.process_data(data, X1, Y1)
Y = torch.from_numpy(Y)
path=saved_path
name=str(0)
ply1, ply2 = convert_ply(X,Y)
dataset="lol"
ext = ".png"
vis = o3d.visualization.Visualizer()
vis.create_window(width = 1080, height= 1080, top = 100, left=50)  

ply1x = ply1.get_rotation_matrix_from_xyz((34/20 * np.pi, 0/20 * np.pi, -12/10 * np.pi))
ply2x = ply2.get_rotation_matrix_from_xyz((34/20 * np.pi, 2/20 * np.pi, -12/10 * np.pi))

ply1.rotate(ply1x, center=(0,0,0))
ply2.rotate(ply2x, center=(0,0,0))
vis.get_render_option().point_size = 5  


# ply1x = ply1.get_rotation_matrix_from_xyz((0/10 * np.pi,0*np.pi,0))
# ply2x = ply2.get_rotation_matrix_from_xyz((0/10 * np.pi, 0*np.pi,0))
# ply1.rotate(ply1x, center=(0,0,0))
# ply2.rotate(ply2x, center=(0,0,0))


vis.add_geometry(ply1, reset_bounding_box = True)
vis.add_geometry(ply2, reset_bounding_box = True)
#o3d.visualization.draw_geometries([ply1, ply2])

vis.get_render_option().point_size = 5  
vis.get_view_control().translate(10, 30, 0, 0)
vis.get_render_option().background_color = [211/255, 211/255, 211/255]
o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.25)

vis.capture_screen_image(path + name + ext, do_render= True)
vis.destroy_window()



#dragon
# ply1x = ply1.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0))
# ply2x = ply2.get_rotation_matrix_from_xyz((1.1/5 * np.pi,0*np.pi,0))
# ply1.rotate(ply1x, center=(0,0,0))
# ply2.rotate(ply2x, center=(0,0,0))
# vis.get_render_option().point_size = 8  
# vis.get_view_control().translate(10, 50, 0, 0)

#o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.2)

#bunny 
#ply1x = ply1.get_rotation_matrix_from_xyz((4/20 * np.pi,0,0))
#ply2x = ply2.get_rotation_matrix_from_xyz((4/20 * np.pi, 0,0))
#ply1.rotate(ply1x, center=(0,0,0))
#ply2.rotate(ply2x, center=(0,0,0))
#vis.get_render_option().point_size = 5  
#vis.get_view_control().translate(10, 50, 0, 0)
#o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.2)


# mubble 
#ply1x = ply1.get_rotation_matrix_from_xyz((0/10 * np.pi,0*np.pi,0))
#ply2x = ply2.get_rotation_matrix_from_xyz((0/10 * np.pi, 0*np.pi,0))
#vis.get_view_control().translate(30, 10, 0, 0)
#o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.38)

#witchcastle 
#ply1x = ply1.get_rotation_matrix_from_xyz((34/20 * np.pi, 0/20 * np.pi, -12/10 * np.pi))
#ply2x = ply2.get_rotation_matrix_from_xyz((34/20 * np.pi, 2/20 * np.pi, -12/10 * np.pi))
#vis.get_view_control().translate(10, 30, 0, 0)
#o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.25)

#hlp.generate(X, Y, saved_path, "0")

# vis = o3d.visualization.Visualizer()
# #vis.create_window(width = 1080, height= 1080, top = 100, left=100)
# ply1x = ply1.get_rotation_matrix_from_xyz((6/20 * np.pi,0 * np.pi,0 * np.pi))
# ply2x = ply2.get_rotation_matrix_from_xyz((0 * np.pi,0 * np.pi,0 * np.pi))
# ply1.rotate(ply1x, center=(0,0,0))
# ply2.rotate(ply2x, center=(0,0,0))

# vis.add_geometry(ply1, reset_bounding_box = True)
# vis.add_geometry(ply2, reset_bounding_box = False)
# vis.get_render_option().point_size = 5  
# vis.get_view_control().translate(30, 0, 0, 0)
# vis.get_render_option().background_color = [211/255, 211/255, 211/255]
# o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.15)
# vis.capture_screen_image(path + name + ext, do_render= True)
# vis.destroy_window()