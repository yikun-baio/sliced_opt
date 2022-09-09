#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 06:48:59 2022

@author: baly
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:47:09 2022

@author: baly
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

method='/sopt'

item_list=['/stanford_bunny','/dragon','/mumble_sitting','/witchcastle']


label_L=['0','1','2','3']
L=['/10k','/9k','/8k','/7k']


#extract the grand truth
item=item_list[2]
time_list=torch.load('experiment/shape_registration/result'+item+'time_list.pt')
print(item)
timepoint=np.array([60, 180, 300])
for key in time_list:
    print(key)
    List=time_list[key]
    for method in List:
        print(method)
        print(List[method]['per_time'])
        index=(timepoint/List[method]['per_time']).astype(np.int64)
        print('pick index',index)




        
   

