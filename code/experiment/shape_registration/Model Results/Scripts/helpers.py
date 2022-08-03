import open3d as o3d
import numpy as np
import os
import torch


def load_data(image_path, ext = ".pt") : 
    data = torch.load(image_path + ext) 
    return data

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

def generate_o3d(ply1, ply2, path, name, dataset="lol", ext = ".jpg"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1080, height= 1080, top = 100, left=50)  
    # ply1x = ply1.get_rotation_matrix_from_xyz((4/20 * np.pi,0,0))
    # ply2x = ply2.get_rotation_matrix_from_xyz((4/20 * np.pi, 0,0))
    # ply1.rotate(ply1x, center=(0,0,0))
    # ply2.rotate(ply2x, center=(0,0,0))
    
    vis.add_geometry(ply1, reset_bounding_box = True)
    vis.add_geometry(ply2, reset_bounding_box = False)
    vis.get_render_option().point_size = 2.25  
    vis.get_view_control().translate(30, 0, 0, 0)
    vis.get_render_option().background_color = [211/255, 211/255, 211/255]
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.37)
    vis.capture_screen_image(path + name + ext, do_render= True)
    vis.destroy_window()
    # vis.run()



# Stanford Bunny 
    # ply1x = ply1.get_rotation_matrix_from_xyz((4/20 * np.pi,0,0))
    # ply2x = ply2.get_rotation_matrix_from_xyz((4/20 * np.pi, 0,0))
    # ply1.rotate(ply1x, center=(0,0,0))
    # ply2.rotate(ply2x, center=(0,0,0))


# Dragon11
    # ply1x = ply1.get_rotation_matrix_from_xyz((4/20 * np.pi,0,0))
    # # ply2x = ply2.get_rotation_matrix_from_xyz((2/20 * np.pi, 0,0))
    # ply1.rotate(ply1x, center=(0,0,0))
    # # ply2.rotate(ply2x, center=(0,0,0))


# Witch castle
    # ply1x = ply1.get_rotation_matrix_from_xyz((34/20 * np.pi, 0/20 * np.pi, -12/10 * np.pi))
    # ply2x = ply2.get_rotation_matrix_from_xyz((34/20 * np.pi, 2/20 * np.pi, -12/10 * np.pi))
    # ply1.rotate(ply1x, center=(1,1,1))
    # ply2.rotate(ply2x, center=(0,0,0))

def generate(X, Y, save_path, name):
    X_ply, Y_ply = convert_ply(X,Y)
    generate_o3d(X_ply, Y_ply, save_path, name)
    

    
    
    