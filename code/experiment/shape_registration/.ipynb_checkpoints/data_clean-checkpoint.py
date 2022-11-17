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

save_root=root+'/saved'
#

    
item='/stanford_bunny'



path=root+'/'+item

pcd = o3d.io.read_point_cloud(path)
data0=np.float32(np.asarray(pcd.points))






    
    
    
    
