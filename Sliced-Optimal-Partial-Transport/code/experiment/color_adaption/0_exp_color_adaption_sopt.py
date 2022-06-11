
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:57:37 2022

@author: laoba
"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread,imsave
    from skimage.segmentation import slic
    from sklearn.cluster import KMeans
    import torch
    import torch.optim as optim
    from skimage.segmentation import slic
    import os
    parent_path='G:/My Drive/Github/Yikun-Bai/Sliced-Optimal-Partial-Transport/Code/'
    dataset_path =parent_path+'dataset/'
    Image_path = dataset_path+'Images/'
    os.chdir(parent_path)
    from library import *
    from sliced_opt import *   
#    from pooling import *
#    from pooling2 import *
#    from develop import *
    from torch import optim
    
    
    
    
    # load picture and reduce the resolution 
    
    Image_s=imread(Image_path+'/source.jpg')
    Image_t=imread(Image_path+'/target.jpg')
    
    n_clusters=500
    M1,N1,C=Image_s.shape
    M2,N2,C=Image_t.shape
    
    data_s=Image_s.reshape(-1,C)
    data_t=Image_t.reshape(-1,C)
     
    kmeans=torch.load('experiment/color_adaption/data/500mean.pt')
    kmeans_s=kmeans['source']
    kmeans_t=kmeans['target']

    label_s=kmeans_s.predict(data_s)
    centroid_s=kmeans_s.cluster_centers_
    centroid_t=kmeans_t.cluster_centers_
    device='cpu'
    dtype=torch.float
#    Lambdalist=[15,18,20]
    #device = "cuda" if torch.cuda.is_available() else "cpu"
  
 #   for Lambda in Lambdalist:
  #      print('Lambda is',Lambda)
    Lambda_list=[110,120]
    for Lambda in Lambda_list:
    # Lambda=20
        centroid_s_T=None
        optimizer=None
        
        del centroid_s_T
        del optimizer
        #mass_pre=430
        
        centroid_s_T=torch.tensor(centroid_s,device=device,requires_grad=True,dtype=dtype)
        centroid_t_T=torch.tensor(centroid_t,device=device,dtype=dtype)
        error2_list=[]
        nb_iter_max=400
        n_projections=C*15
        optimizer=optim.Adam([centroid_s_T],lr=0.1,weight_decay=0)
        print('Lambda',Lambda)
        for epoch in range(0,nb_iter_max):
            optimizer.zero_grad()
            A=sopt(centroid_s_T,centroid_t_T,Lambda,n_projections,'orth')
            loss,mass=A.sliced_cost()
            loss.backward() 
            if epoch%10==0:
                print('training Epoch {}/{}'.format(epoch, nb_iter_max))
                print('gradient is',torch.norm(centroid_s_T.grad).item())
                print('loss is ',loss.item())
                print('mass is',mass)
                print('-' * 10)
            optimizer.step()

            grad_norm=torch.norm(centroid_s_T.grad)
            if grad_norm>=20:
                optimizer.param_groups[0]['lr']=2
            elif grad_norm>=10:
                optimizer.param_groups[0]['lr']=1
            elif grad_norm>=5:
                optimizer.param_groups[0]['lr']=1
            elif grad_norm>=1:
                optimizer.param_groups[0]['lr']=0.5
            else:
                break

        
    
     
        centroid_s_final_opt=centroid_s_T.clone().detach().cpu()
        centroid_s_final_opt1=torch.tensor(centroid_s,device=device,dtype=dtype)
        cost_M=cost_matrix_T(centroid_s_final_opt,centroid_t_T)
        Delta=10
        n_point=0
        for i in range(500):
            if torch.sum(cost_M[i,:]<=Delta)>=1:
                centroid_s_final_opt1[i,:]=centroid_s_final_opt[i,:]
                n_point+=1
        
        torch.save(centroid_s_T,'experiment/color_adaption/model/centroid_s_T'+str(Lambda)+'.pt')
        print('transfer {} colors'.format(n_point))
        centroid_s_final_opt1=centroid_s_final_opt1.numpy()

        Image_s_opt=centroid_s_final_opt1[label_s,:].reshape(M1,N1,C)
    
        fig,ax=plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(Image_s.astype(np.uint8))
        ax[1].imshow(Image_s_opt.astype(np.uint8))
        plt.show()
        #torch.save(Image_s_opt,'experiment/color_adaption/model/image_s_pos_+Lambda'+str(Lambda)+'.pt')
        #imsave('experiment/color_adaption/result/max_'+str(Lambda)+'.jpg',Image_s_opt.astype(np.uint8))

      
    