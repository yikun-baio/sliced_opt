#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:36:23 2022

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
work_path=os.path.dirname(__file__)
print('work_path is', work_path)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.lib_shape import *



root = parent_path+'/experiment/shape_registration/data/test2'
#save_root = os.path.join(root+'/testdata', dataset_name)
save_root=root
if not os.path.exists(save_root):
    os.makedirs(save_root)

from experiment.shape_registration.dataset import Dataset


    
item='castle'

def load(path):
    """takes as input the path to a .pts and returns a list of 
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]
    n=len(rows)-1
    L=[]
    for i in range(1,n+1):
        row=rows[i].split(' ')[1:]
        d=len(row)
        row_d=np.zeros(d)
        for j in range(0,d):
            row_d[j]=float(row[j])
        L.append(row_d)
    L=np.array(L)
    matrix=L.reshape(n,d)
    return matrix

data=load(root+'/WitchCastle_150000.txt')


data=torch.from_numpy(data)
data=data.to(torch.float32)
n,d=data.shape
N=10*int(1e3)
randint=torch.randint(0,n,(N,))
X0=data[randint]
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X0[:,0],X0[:,1],X0[:,2],s=0.03) # plot the point   (2,3,4) on the figure
plt.show()
theta=torch.tensor([1/5*torch.pi,1/5*torch.pi,1/4*torch.pi])
rotation=rotation_3d_2(theta,'in')
scalar=0.5 #0.6
beta=torch.tensor([0,0,0],dtype=torch.float32) #torch.tensor([1.8,0.5,0.5])
Y0=scalar*X0@rotation+beta
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X0[:,0],X0[:,1],X0[:,2],s=0.03,label='target') # plot the point (2,3,4) on the figure
ax.scatter(Y0[:,0],Y0[:,1],Y0[:,2],s=0.03,label='source') # plot the point (2,3,4) on the figure
plt.legend()
plt.show()
plt.close()
data={}

N1=5*int(1e3)
N2=8*int(1e3)
N3=9*int(1e3)

randint=torch.randint(0,N,(N1,))
Y01=Y0[randint]
randint=torch.randint(0,N,(N2,))
Y02=Y0[randint]
randint=torch.randint(0,N,(N3,))
Y03=Y0[randint]

#N4=10*int(1e3)

randint=torch.randint(0,n,(N1,))
Y01=Y0
data['X0']=X0
data['Y0']=Y0
data['Y01']=Y01
data['Y02']=Y02
data['Y03']=Y03

data['theta']=theta
data['beta']=beta
data['scalar']=scalar
# test if the data is symmetric 
#recover_rotation(X,Y)

torch.save(data,save_root+'/data'+str(item)+'.pt')

#X1T_c=X-torch.mean(X,0)
#Y1T_c=Y-torch.mean(Y,0)
#U1,S1,V1=torch.pca_lowrank(X1T_c)
#U2,S2,V2=torch.pca_lowrank(Y1T_c)



    
    
    
    
