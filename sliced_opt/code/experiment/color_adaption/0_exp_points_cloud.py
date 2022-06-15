# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:11:12 2022

@author: laoba
"""
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_swiss_roll
    import torch
    from torch import optim
    
    import ot
    import os
    parent_path='G:/My Drive/Github/Yikun-Bai/Sliced-Optimal-Partial-Transport/Code/'
    os.chdir(parent_path)
    dataset_path =parent_path+'datasets/'
    Image_path = dataset_path+'Images/'
    
    
    from library import *
    from sliced_opt import *
    
    label='1'

    def rotation_matrix(theta):
          return torch.stack([torch.cos(theta).reshape([1]),torch.sin(theta).reshape([1]),
                -torch.sin(theta).reshape([1]),torch.cos(theta).reshape([1])]).reshape([2,2])

    def scalar_matrix(scalar):
      return torch.stack([scalar[0:2],scalar[1:3]])
    
    def scalar_matrix(scalar):
      return torch.stack([scalar[0:2],scalar[1:3]])
    
    X0_T=torch.load('experiment/point_cloud_matching/data/data'+label+'/X0_T.pt')
    X1_T=torch.load('experiment/point_cloud_matching/data/data'+label+'/X1_T.pt')
    Y0_T=torch.load('experiment/point_cloud_matching/data/data'+label+'/Y0_T.pt')
    Y1_T=torch.load('experiment/point_cloud_matching/data/data'+label+'/Y1_T.pt')
    
    fig=plt.figure(figsize=(5,5))
    #plt.scatter(X00[:,0],X00[:,1],label='target0')
    plt.scatter(Y1_T[:,0],Y1_T[:,1],label='source')
    plt.scatter(X1_T[:,0],X1_T[:,1],label='target')
    plt.legend()
    plt.show()

    parameter=torch.load('experiment/point_cloud_matching/data/data'+label+'/parameter.pt')
    op_theta=parameter['op_theta']
    op_scalar=parameter['op_scalar']
    op_beta=parameter['op_beta']
    N=120
    N_noise=10
    margin1=7.5
    margin2=7.5
    n_iteration=800
    
        
    
    #sopt 



    device='cpu'
    dtype=torch.float
    Lambda=11
    Delta=0.2
    threashold=4
    
    theta=torch.tensor(0,dtype=dtype,requires_grad=True,device=device)
    beta=torch.tensor([0,0],dtype=dtype,requires_grad=True,device=device)
    scalar=torch.tensor([1,0,1],dtype=dtype,requires_grad=True,device=device)
    n_projections=80
    # task 1 [0.15,0.5,0.15] 0.01
    # task 2 [0.2,0.6,0.2] 0.01 
    # task 3 []
    # task 5 [0.4,0.6,0.4] 0.01
    optimizer1=optim.Adam([scalar],lr=0.15,weight_decay=0.01)
    optimizer2=optim.Adam([beta],lr=0.5,weight_decay=0.01)
    optimizer3=optim.Adam([theta],lr=0.15,weight_decay=0.01)
    mu=np.ones([N+N_noise])
    nu=np.ones([N+N_noise])
    mu0=1/N*np.ones(N)
    nu0=1/N*np.ones(N)
    
    errorlist_sopt=[]
    
    # compute the parameter error
    parameterlist=[]

    
    for epoch in range(n_iteration):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        
        rotation=rotation_matrix(theta)
        scalarM=scalar_matrix(scalar)
        X1_hat_T=Y1_T@scalarM@rotation+beta
        
        A=sopt(X1_hat_T,X1_T,Lambda,n_projections,'orth')
        loss,n_point=A.sliced_cost()
        loss.backward()
        if n_point>N*1.003:
          Lambda-=Delta
        elif n_point<N*0.98:
          Lambda+=Delta
        
        if Lambda<=threashold:
          Lambda=threashold
          threashold=threashold/2
          Delta=Delta/2
        
        
        # compute the error
        rotation_np=rotation.clone().detach().cpu().numpy()
        scalarM_np=scalarM.clone().detach().cpu().numpy()
        beta_np=beta.clone().detach().cpu().numpy()
        X0_hat = Y0_T.cpu().numpy()@scalarM_np@rotation_np+beta_np[np.newaxis,:]
        M0=cost_matrix(X0_T.cpu().numpy(),X0_hat)
        W2_error=ot.emd2(mu0, nu0, M0)
        errorlist_sopt.append(W2_error)
        
        # compute the parameter error
        prarameter={}
        parameter['theta']=theta
        parameter['scalar']=scalar
        parameter['beta']=beta
        parameterlist.append(parameter)
        
        #errorlist_sopt_theta.append(np.linalg.norm((theta.clone().detach().cpu()-op_theta)))
        #errorlist_sopt_scalar.append(np.linalg.norm(scalar.clone().detach().cpu()-op_scalar))
        #errorlist_sopt_beta.append(np.linalg.norm(beta.clone().detach().cpu()-op_beta))
        
        
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()



        
        if epoch%40==0 or epoch==n_iteration-1:
          X1_hat_T1=X1_hat_T.detach().cpu()
          fig=plt.figure(figsize=(5,5))
          plt.scatter(X1_hat_T1[:,0],X1_hat_T1[:,1],label='source')
          plt.scatter(X1_T[:,0],X1_T[:,1],label='target')
          plt.xlim(-margin1,margin1)
          plt.ylim(-margin2,margin2)
          plt.legend(loc='upper right')
          plt.savefig('experiment/point_cloud_matching/result/sopt/data'+str(label)+'/'+str(epoch)+'.jpg')
          plt.show()
        
          print('Lambda is',Lambda)
          print('n_point is',n_point)
          print('delta is',Delta)
          print('training Epoch {}/{}'.format(epoch, n_iteration))
          print('grad of theta is',theta.grad.item())
          print('grad of scalar matrix is',torch.norm(scalar.grad))
          print('grad of shift is',torch.norm(beta.grad))
          print('loss is ',loss.item())
          print('-' * 10)
        if epoch%40==0:
          parameter={'theta':theta,'scalar':scalar,'beta':beta}
          torch.save(parameter,'experiment/point_cloud_matching/model/sopt/'+str(epoch)+'parameter.pt')
        
    torch.save(errorlist_sopt,'experiment/point_cloud_matching/result/sopt/data'+str(label)+'/parameter_list.pt')


   


