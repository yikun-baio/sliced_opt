
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:57:37 2022

@author: laoba
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

work_path=os.path.dirname(__file__)
# load the data 
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import ot


rng = np.random.RandomState(42)



exp_num='1'
data_path='experiment/color_adaption/data/'+exp_num
I1=imread(data_path+'/source.jpg').astype(np.float64) / 256
I2=imread(data_path+'/target.jpg').astype(np.float64) / 256

M1,N1,C=I1.shape
M2,N2,C=I2.shape
X1=I1.reshape(-1,C)
X2=I2.reshape(-1,C)
rng = np.random.RandomState(42)
nb1 = 1000
nb2 = 2000
idx1 = rng.randint(X1.shape[0], size=(nb1,))
idx2 = rng.randint(X2.shape[0], size=(nb2,))
start_time=time.time()
Xs = X1[idx1, :]
Xt = X2[idx2, :]

start_time=time.time()


XsT=torch.from_numpy(Xs).to(dtype=torch.float)
XtT=torch.from_numpy(Xt).to(dtype=torch.float)

n_projections=1000
Lambda=10.0
Lambda_list=torch.full((n_projections,),Lambda)
A=sopt_correspondence(XsT.clone(),XtT.clone(),Lambda_list,n_projections)
A.n_projections=600
A.get_directions()
A.X=XsT.clone()
A.Y=XtT.clone()
A.correspond()
transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))
end_time=time.time()
wall_time=end_time-start_time
I1t = minmax(mat2im(transp_Xs, I1.shape))
plt.figure(figsize=(10,10))

plt.axis('off')
#    plt.title('Image Source')
plt.imshow(I1t)
plt.savefig('experiment/color_adaption/results/'+exp_num+'/sopt'+str(Lambda)+'.png',format='png',dpi=2000)
plt.close()

result={}
result['transp_Xs']=transp_Xs
result['time']=wall_time
torch.save(result,'experiment/color_adaption/results/'+exp_num+'/sopt'+str(Lambda)+'.pt')



# A.X=XsT.clone()
# A.Y=XtT.clone()
# A.Lambda_list=torch.full((A.n_projections,),np.float32(5))
# A.correspond()
# transp_Xs=A.transform(torch.from_numpy(X1).to(dtype=torch.float32))

# I1t = minmax(mat2im(transp_Xs, I1.shape))
# plt.figure(figsize=(10,10))

# plt.axis('off')
# #    plt.title('Image Source')
# plt.imshow(I1t)
# plt.savefig('experiment/color_adaption/results/'+exp_num+'/spot.jpg')
# plt.close()
# torch.save(transp_Xs,'experiment/color_adaption/results/'+exp_num+'/spot'+'.pt')
#plt.show()
# # EMDTransport
# ot_emd = ot.da.EMDTransport()
# ot_emd.fit(Xs=Xs, Xt=Xt)

# # SinkhornTransport
# ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-1)
# ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

# # prediction between images (using out of sample prediction as in [6])
# transp_Xs_emd = ot_emd.transform(Xs=X1)
# transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)
# print('Xs A',Xs[0:10])
# transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X1)
# transp_Xt_sinkhorn = ot_sinkhorn.inverse_transform(Xt=X2)

# I1t = minmax(mat2im(transp_Xs_sopt, I1.shape))
# # I2t = minmax(mat2im(transp_Xt_emd, I2.shape))

# # I1te = minmax(mat2im(transp_Xs_sinkhorn, I1.shape))
# # I2te = minmax(mat2im(transp_Xt_sinkhorn, I2.shape))

# plt.figure(2, figsize=(10, 8))
# plt.subplot(2, 2, 1)
# plt.imshow(I1)
# plt.axis('off')
# plt.title('Image Source')

# plt.subplot(2, 2, 2)
# plt.imshow(I2)
# plt.axis('off')
# plt.title('Image Target')


# plt.subplot(2, 2, 3)
# plt.imshow(I1)
# plt.axis('off')
# plt.title('Image Source')

# plt.subplot(2, 2, 4)
# plt.imshow(I1t)
# plt.axis('off')
# plt.title('Image Trans by sopt')


# plt.subplot(2, 3, 3)
# plt.imshow(I1te)
# plt.axis('off')
# plt.title('Image 1 Adapt (reg)')

# plt.subplot(2, 3, 4)
# plt.imshow(I2)
# plt.axis('off')
# plt.title('Image 2')

# plt.subplot(2, 3, 5)
# plt.imshow(I2t)
# plt.axis('off')
# plt.title('Image 2 Adapt')

# plt.subplot(2, 3, 6)
# plt.imshow(I2te)
# plt.axis('off')
# plt.title('Image 2 Adapt (reg)')
# plt.tight_layout()

# plt.show()


# n_clusters=2000
# kmeans=torch.load('experiment/color_adaption/data/task'+task_num+'/kmean'+str(n_clusters)+'.pt')
# kmean1=kmeans['source']
# kmean2=kmeans['target']

# label1=kmean1.predict(data1)
# centroid1=kmean1.cluster_centers_.astype(np.float32)
# centroid2=kmean2.cluster_centers_.astype(np.float32)
# device='cpu'
# dtype=torch.float

# #N=1500
# #Lambda=0.3
# X1=torch.tensor(centroid1,device=device,dtype=dtype)
# X2=torch.tensor(centroid2,device=device,dtype=dtype)
# error2_list=[]
# nb_iter_max=3000
# n_projections=C*15
# Lambda=30
# Delta=Lambda*1/10
# print('Lambda',Lambda)
# A=sopt(X1,X2,Lambda,nb_iter_max,'orth')


# for epoch in range(0,nb_iter_max):
#     A.get_one_projection(epoch)
#     A.get_plans()
#     loss,mass=A.sliced_cost()
#    # mass_diff=mass.item()-N
#     n=A.X_take.shape[0]
#     A.X[A.Lx]+=(A.Y_take-A.X_take).reshape((n,1))*A.projections[epoch]
    
    
#     # if mass_diff>N*0.012:
#     #     A.Lambda-=Delta
    
#     # if mass_diff<-N*0.003:
#     #     A.Lambda+=Delta
#     # if A.Lambda<=Delta:
#     #     A.Lambda=Delta
#     #     Delta=Delta/2
#     if epoch<=50 or epoch%10==0:
#         print('mass',mass)
# #        print('lambda',A.Lambda)


    
    

# centroid1_f=A.X.clone().detach().cpu().numpy()
# centroid1_f2=centroid1
# cost_M=cost_matrix_d(centroid1_f,centroid2)
# Delta=10
# n_point=0
# for i in range(n_clusters):
#     if np.sum(cost_M[i,:]<=Delta)>=1:
#         centroid1_f2[i,:]=centroid1_f[i,:]
#         n_point+=1

# torch.save(centroid1_f,'experiment/color_adaption/results/sopt'+str(Lambda)+'.pt')
# torch.save(centroid1_f2,'experiment/color_adaption/results/sopt'+str(Lambda)+'.pt')

# print('transfer {} colors'.format(n_point))
# I1_f2=centroid1_f2[label1,:].reshape(M1,N1,C)
# I1_f=centroid1_f[label1,:].reshape(M1,N1,C)
# fig,ax=plt.subplots(1,3,figsize=(15,5))
# ax[0].imshow(I1.astype(np.uint8))
# ax[1].imshow(I1_f.astype(np.uint8))
# ax[2].imshow(I1_f2.astype(np.uint8))
# plt.show()
# imsave('experiment/color_adaption/results/sopt'+str(Lambda)+'.jpg',I1_f.astype(np.uint8))

  
