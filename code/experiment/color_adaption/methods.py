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




def recover_image(transp_Xs,shape,name,save_path):
    #print(transp_Xs)
    I1t = minmax(mat2im(transp_Xs, shape))
    plot_image(I1t,name,save_path)


def plot_image(I1t,name,save_path):
    #print(transp_Xs)
    plt.figure()
    plt.axis('off')
    plt.imshow(I1t)
    #plt.pad_inces=0.01
    plt.savefig(save_path+'/'+name+'.png',format="png",dpi=800,bbox_inches='tight',pad_inches = 0)
    plt.show()
    #f.clear()
    plt.close()
    
def ot_transfer(X1,X2,Xs,Xt):
#    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
#    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    # EMDTransport
    ot_emd = ot.da.EMDTransport(max_iter=500000)
    
    ot_emd.fit(Xs=Xs, Xt=Xt)
    # # prediction between images (using out of sample prediction as in [6])
    transp_Xs = ot_emd.transform(Xs=X1)
    return transp_Xs
#   transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)


def eot_transfer(X1,X2,Xs,Xt):
    # SinkhornTransport
    ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1,max_iter=500000)
    ot_sinkhorn.fit(Xs=Xs, Xt=Xt)
    transp_Xs = ot_sinkhorn.transform(Xs=X1)
    return transp_Xs
    

def spot_transfer(X1,X2,Xs,Xt,n_projections=300):
    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    A=spot(XsT.clone(),XtT.clone(),n_projections)
    A.get_directions()
    A.correspond()
    transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))
    return transp_Xs
        
def sopt_transfer(X1,X2,Xs,Xt,Lambda_list,n_projections):
    XsT=torch.from_numpy(Xs).to(dtype=torch.float)
    XtT=torch.from_numpy(Xt).to(dtype=torch.float)
    A=sopt_correspondence(XsT.clone(),XtT.clone(),Lambda_list,n_projections)
    A.get_directions()
    A.correspond()
    transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))
    return transp_Xs

exp_num_list=[0,1,2,3]
number_list=[(1,1),(1,2),(2,1),(2,2)]
exp_num =2

s_n,t_n=number_list[exp_num]

data_path='experiment/color_adaption/data/'+str(exp_num)
save_path='experiment/color_adaption/results/'+str(exp_num)
I1=imread(data_path+'/source'+str(s_n)+'.jpg').astype(np.float64) / 256
I2=imread(data_path+'/target'+str(t_n)+'.jpg').astype(np.float64) / 256

M1,N1,C=I1.shape
M2,N2,C=I2.shape
X1=I1.reshape(-1,C)
X2=I2.reshape(-1,C)

N1=5000
N2=10000
try:
    kmean_X1=torch.load(data_path+'/kmeans_S'+str(s_n)+'_'+str(N1)+'.pt')
except:
    rng = np.random.RandomState(42)
    idx1 = rng.randint(X1.shape[0], size=(N1,))
    
kmean_X2=torch.load(data_path+'/kmeans_T'+str(t_n)+'_'+str(N2)+'.pt')
# Xs = kmean_X1.cluster_centers_
Xt = kmean_X2.cluster_centers_

# idx2 = rng.randint(X2.shape[0], size=(nb2,))

Xs = X1[idx1, :]

# method_list=['ot','eot','spot','sopt1.0','sopt0.1']
plot_image(I1,'source',save_path)
plot_image(I2,'target',save_path)

# for method in method_list:
#     transp_Xs=torch.load(save_path+'/'+method+'.pt')
#     name=method
#     recover_image(transp_Xs,I1.shape,name,save_path)
    
#time_list={}
try:
    time_list=torch.load(save_path+'/time_list.pt')
except: 
    time_list={}
                                                                                 
start_time=time.time()
transp_Xs=eot_transfer(X1,X2,Xs,Xt)
end_time=time.time()

wall_time=end_time-start_time
time_list['eot']=wall_time

recover_image(transp_Xs,I1.shape,'/eot',save_path)
torch.save(transp_Xs,save_path+'/eot.pt')
torch.save(wall_time,save_path+'/eot_time.pt')


start_time=time.time()
transp_Xs=ot_transfer(X1,X2,Xs,Xt)
end_time=time.time()
wall_time=end_time-start_time 
time_list['ot']=wall_time
recover_image(transp_Xs,I1.shape,'/ot',save_path)
torch.save(transp_Xs,save_path+'/ot.pt')
torch.save(wall_time,save_path+'/ot_time.pt')

print('spot')
n_projections=300
start_time=time.time()
transp_Xs=spot_transfer(X1,X2,Xs,Xt,n_projections)
end_time=time.time()
wall_time=end_time-start_time
time_list['spot']=wall_time 
torch.save(transp_Xs,save_path+'/spot.pt')
torch.save(wall_time,save_path+'/spot_time.pt')
recover_image(transp_Xs,I1.shape,'spot',save_path)

print('sopt')

n_projections=300
Lambda=np.float32(1)
Lambda_list=torch.full((n_projections,),Lambda)
start_time=time.time()
transp_Xs=sopt_transfer(X1,X2,Xs,Xt,Lambda_list,n_projections)
end_time=time.time()
wall_time=end_time-start_time 
time_list['sopt'+str(Lambda)]=wall_time 
torch.save(transp_Xs,save_path+'/sopt'+str(Lambda)+'.pt')
torch.save(wall_time,save_path+'/sopt'+str(Lambda)+'_time.pt')
recover_image(transp_Xs,I1.shape,'/sopt'+str(Lambda),save_path)


n_projections=300
Lambda=np.float32(0.001)
Lambda_list=torch.full((n_projections,),Lambda)
start_time=time.time()
transp_Xs=sopt_transfer(X1,X2,Xs,Xt,Lambda_list,n_projections)
end_time=time.time()
wall_time=end_time-start_time 
time_list['sopt'+str(Lambda)]=wall_time 
torch.save(transp_Xs,save_path+'/sopt'+str(Lambda)+'.pt')
torch.save(wall_time,save_path+'/sopt'+str(Lambda)+'_time.pt')
recover_image(transp_Xs,I1.shape,'/sopt'+str(Lambda),save_path)
torch.save(time_list,save_path+'/time_list.pt')