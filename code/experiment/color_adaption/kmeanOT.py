#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:29:08 2022

@author: baly
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread,imsave
from skimage.segmentation import slic
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
from skimage.segmentation import slic
import os
import sys
import ot
import time
work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


from sopt2.library import *
from sopt2.sliced_opt import *   
from sopt2.lib_color import *   
from torch import optim






def show_figure(transp_Xs,shape,name,save_path):
    I1t = minmax(mat2im(transp_Xs, shape))
    plt.figure() #figsize=(10,10))
    plt.axis('off')
    plt.imshow(I1t)
    plt.savefig(save_path+name,format="png",dpi=2000)
    plt.close()

def kmean_ot(X1,X2,Xs,Xt):
#    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
#    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    # EMDTransport
    ot_emd = ot.da.EMDTransport()
    ot_emd.fit(Xs=Xs, Xt=Xt)
    # # prediction between images (using out of sample prediction as in [6])
    transp_Xs = ot_emd.transform(Xs=X1)
    return transp_Xs
#   transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)


def kmean_eot(X1,X2,Xs,Xt):
    # SinkhornTransport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    transp_Xs = ot_sinkhorn.transform(Xs=X1)
    return transp_Xs
    

def kmean_spot(X1,X2,Xs,Xt,n_projections=500):
    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    A=spot(XsT.clone(),XtT.clone(),n_projections)
    A.get_directions()
    A.correspond()
    transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))
    return transp_Xs
        
def kmean_sopt(X1,X2,Xs,Xt,n_projections,Lambda_list):
    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    A=sopt_correspondence(XsT.clone(),XtT.clone(),Lambda_list,n_projections)
    A.get_directions()
    A.correspond()
    transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))
    return transp_Xs

exp_num='1'
data_path='experiment/color_adaption/data/'+exp_num
save_path='experiment/color_adaption/results/'+exp_num
I1=imread(data_path+'/source.jpg').astype(np.float64) / 256
I2=imread(data_path+'/target.jpg').astype(np.float64) / 256

M1,N1,C=I1.shape
M2,N2,C=I2.shape
X1=I1.reshape(-1,C)
X2=I2.reshape(-1,C)

N1=5000
N2=10000
kmean_X1=torch.load(data_path+'/kmeans_X1_'+str(N1)+'.pt')
kmean_X2=torch.load(data_path+'/kmeans_X2_'+str(N2)+'.pt')
Xs = kmean_X1.cluster_centers_
Xt = kmean_X2.cluster_centers_


# start_time=time.time()
# transp_Xs=kmean_eot(X1,X2,Xs,Xt)
# end_time=time.time()
# wall_time=end_time-start_time 
# show_figure(transp_Xs,I1.shape,'/eot.png',save_path)
# torch.save(transp_Xs,save_path+'/eot.pt')
# torch.save(wall_time,save_path+'/eot_time.pt')


start_time=time.time()
transp_Xs=kmean_ot(X1,X2,Xs,Xt)
end_time=time.time()
wall_time=end_time-start_time 
show_figure(transp_Xs,I1.shape,'/ot.png',save_path)
torch.save(transp_Xs,save_path+'/ot.pt')
torch.save(wall_time,save_path+'/ot_time.pt')


# start_time=time.time()
# transp_Xs=kmean_spot(X1,X2,Xs,Xt,600)
# end_time=time.time()
# wall_time=end_time-start_time 
# torch.save(transp_Xs,save_path+'/spot.pt')
# torch.save(wall_time,save_path+'/spot_time.pt')
# show_figure(transp_Xs,I1.shape,'/spot.png',save_path)

# #n_projections=400
# Lambda=np.float32(10)
# Lambda_list=torch.full((n_projections,),Lambda)

# n_projections=600
# start_time=time.time()
# transp_Xs=kmean_sopt(X1,X2,Xs,Xt,n_projections,Lambda_list)
# end_time=time.time()
# wall_time=end_time-start_time 

# torch.save(transp_Xs,save_path+'/sopt'+str(Lambda)+'.pt')
# torch.save(wall_time,save_path+'/sopt_time+'+str(Lambda)+'.pt')
# show_figure(transp_Xs,I1.shape,'/sopt_'+str(Lambda)+'.png',save_path)
