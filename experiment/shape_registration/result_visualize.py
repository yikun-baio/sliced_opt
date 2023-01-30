#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:19:27 2022

@author: baly
"""




import sys
import open3d as o3d
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import numpy as np
import ot
import time
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)
from sopt2.library import *
from sopt2.lib_shape import *
from sopt2.sliced_opt import *   

#/stanford_bunny,
# /dragon, 
#'/mumble_sitting'
#'/witchcastle'
item_list=['/witchcastle',]
item=item_list[0]
method_list=['/sopt','/spot','/icp_du','/icp_umeyama']

label_List=['0','1']
n_point_list=['/10k','/9k']
per_s_list=['-5p','-7p']




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
    return np.array(data_indices),np.array(noise_indices)

def init_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+5,X_data[:,1],X_data[:,2]-10,alpha=.5,c='C2',s=2,marker='o')
    ax.scatter(X_noise[:,0]+5,X_noise[:,1],X_noise[:,2]-10,alpha=.5,c='C2',s=10,marker='o')
    ax.scatter(Y_data[:,0]+5,Y_data[:,1],Y_data[:,2]-10,alpha=0.5,c='C1',s=2,marker='o')
    ax.scatter(Y_noise[:,0]+5,Y_noise[:,1],Y_noise[:,2]-10,alpha=.5,c='C1',s=10,marker='o')
    ax.set_facecolor('black') 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(True)
    ax.axis('off')
    
    # whitch_castle 
    # x+5, z-10
    ax.set_xlim([-38,38])
    ax.set_ylim([-38,38])
    ax.set_zlim([-38,38])
    ax.view_init(45,120)
    
    
    #mumble_sitting 
    # ax.set_xlim([-66,66])
    # ax.set_ylim([-66,66])
    # ax.set_zlim([-66,66])
    # ax.axis('off')
    # ax.view_init(-20,10,'y')

    
    #dragon +bunny     
    # bunny y-0.05,
    # ax.set_xlim([-.25,.25])
    # ax.set_ylim([-.25,.25])
    # ax.set_zlim([-.25,.25])
    # ax.axis('off')
    # ax.view_init( 90, -90)
    

    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    
    

def regular_image(X_data,X_noise,Y_data,Y_noise,image_path,name):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.scatter(X_data[:,0]+3,X_data[:,1],X_data[:,2]-15,alpha=.3,c='C2',s=5,marker='o')
    ax.scatter(X_noise[:,0]+3,X_noise[:,1],X_noise[:,2]-15,alpha=.5,c='C2',s=15,marker='o')
    ax.scatter(Y_data[:,0]+3,Y_data[:,1],Y_data[:,2]-15,alpha=.9,c='C1',s=6,marker='o')
    ax.scatter(Y_noise[:,0]+3,Y_noise[:,1],Y_noise[:,2]-15,alpha=.5,c='C1',s=15,marker='o')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')
    ax.set_facecolor('black') 
    ax.grid(True)
    
    # castle,   
    #x+3, z-15 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])
    ax.set_zlim([-20,20])
    ax.view_init(45,120)
#    ax.view_init(0,10,'y')


    # #mumble_sitting, bunny  
    # y-10
    # ax.set_xlim([-36,36])
    # ax.set_ylim([-36,36])
    # ax.set_zlim([-36,36])
    # ax.view_init(-20,10,'y')
    #ax.view_init( 90, -90)
     
    #dragon, bunny 
    #dragon y-0.1
    #bunny, x+0.02, y-0.1    
    # ax.set_xlim([-.1,.1])
    # ax.set_ylim([-.1,.1])
    # ax.set_zlim([-.1,.1])

    # ax.view_init( 90, -90)
    # fig.set_facecolor('black')
    
    plt.savefig(image_path+'/'+name+'.png',dpi=200,format='png',bbox_inches='tight')
    plt.show()
    plt.close()
    

result_path='experiment/shape_registration/result'
time_list=torch.load(result_path+item+'time_list.pt')
# time_list['/10k-7p']=time_list['/10k--7p']
# time_list['/9k-5p']=time_list['/9k--5p']
# del time_list['/10k--7p'] 
# del time_list['/9k--5p'] 
# time_list['/10k-7p']['icp_du']=time_list['/10k-7p']['icp-du']
# time_list['/10k-7p']['icp_umeyama']=time_list['/10k-7p']['icp-umeyama']
# del time_list['/10k-7p']['icp-du']
# del time_list['/10k-7p']['icp-umeyama']

# time_list['/9k-5p']['icp_du']=time_list['/9k-5p']['icp-du']
# time_list['/9k-5p']['icp_umeyama']=time_list['/9k-5p']['icp-umeyama']
# del time_list['/9k-5p']['icp-du']
# del time_list['/9k-5p']['icp-umeyama']
# torch.save(time_list,result_path+item+'time_list.pt')

for (label,per_s) in [('0','-7p'),('1','-5p')]:
    for method in method_list:

        n_point=n_point_list[int(label)]    
       
        data_path=parent_path+'/experiment/shape_registration/data/test2/saved'
        save_path=result_path+item+n_point+per_s
        image_path='experiment/shape_registration/images'+item+n_point+per_s+method
        data=torch.load(data_path+item+'.pt')
        
        print('label',label)
        print('per_s',per_s)
        print('method',method)
        time_dict={}
        
        X0=data['X0'].to(torch.float32)
        Y0=data['Y0'+label].to(torch.float32)
        X1=data['X1'+per_s].to(torch.float32)
        Y1=data['Y1'+label+per_s].to(torch.float32)
        X=X1.numpy().copy()
        Y=Y1.numpy().copy()
        n=X1.shape[0]
        data_indices_X=range(0,10000)
        noise_indices_X=range(10000,n)
    
        data_indices_Y,noise_indices_Y=get_noise(Y0,Y1)
    
        X_data=X1[data_indices_X]
        X_noise=X1[noise_indices_X]
    
        Y_data=Y1[data_indices_Y]
        Y_noise=Y1[noise_indices_Y]
        
        init_image(X_data,X_noise,Y_data,Y_noise,image_path,'init0')
        regular_image(X_data,X_noise,Y_data,Y_noise,image_path,'init')
       
        
        # load parameter: 
        per_time=time_list[n_point+per_s][method[1:]]['per_time']
        param_list=torch.load(save_path+method+'_param.pt')
        N=len(param_list)
        k_list=[]
        for k in range(1,5):
            k_list.append(int(k*60/per_time))
        k_list.append(N-1)
            
        for k in k_list:
            param=param_list[k]
            rotation=param['rotation']
            beta=param['beta']
            scalar=param['scalar']
            X_hat=Y1@rotation*scalar+beta
            X_hat_data=X_hat[data_indices_Y]
            X_hat_noise=X_hat[noise_indices_Y]
            regular_image(X_data,X_noise,X_hat_data,X_hat_noise,image_path,str(k))
            
            
            
    
      

    
    

    
    
    
    
    
