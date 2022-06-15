# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:42:04 2022

@author: laoba
"""

# if __name__ == '__main__':
#     import sopt

# #     import torch
# #     from opt import *
# #     from pooling2 import *
# #     from sliced_opt import *
# #     X=torch.rand([5,3])
# #     Y=torch.rand([5,3])
# #     Lambda=2
# #     n_projections=6
# #     a,b=max_sopt_pool_T1(X,Y,Lambda,n_projections,n_cpu=4,Type=None)
# #     c,d=max_sopt(X,Y,Lambda,n_projections,Type=None)

#     from sliced_opt import *
#     from pooling2 import *
#     import torch


import torch
import os
import torch.multiprocessing as mp
import tempfile

# def f(value):
#  return value*2

# def run(X):
#  with mp.Pool(processes=3) as pool: 
#   result = pool.map(f, X)
#   # result = torch.tensor(result)
#   print(result)





if __name__ == '__main__':
    work_path=os.getcwd()
    loc1=work_path.find('/code')
    parrent_path=work_path[0:loc1+6]
    #print('work_path',work_path)
    #print('loc1',loc1)
    #os.chdir(parrent_path)
    
    from test1 import *
    print('start')

    
    device='cpu'
    X=torch.tensor([1,2,3]).to(device)
    A=test(X,X)
    #print(A.results)



    # import numpy as np
    # import torch
    # from library import *
    # from develop import *
    # from sliced_opt import *
    # import torch.multiprocessing as mp
    # from pooling2 import *
    # import time
    
    # torch.manual_seed(0)
    # device='cpu'
    # X=1.1*torch.rand([15,2])
    # Y=2+torch.rand([15,2])
    # X=X.to(device).to(device).requires_grad_()
    # Y=Y.to(device).to(device)
    # Lambda=4
    # Type=None
    # n_projections=80
    # time1=time.time()
    
    # A=sopt(X,Y,Lambda,n_projections,Type=None)
    # print('cost1',A.sliced_cost())
    # time2=time.time()
    # print('time',time2-time1)
    
    # time1=time.time()
    # cost2=sopt_orig(X,Y,Lambda,n_projections,Type=None)
    # print('cost2',cost2)
    # time2=time.time()
    # print('time',time2-time1)
    


    #     Y1=torch.rand([n+5,3],device='cpu')

        
    #     timestart=time.time()
    #     sopt_pool(X,Y,Lambda,n_projections,8)
    #     timeend=time.time()
    #     time1=timeend-timestart
 
    #     timestart=time.time()
    #     sopt_po(X1,Y1,Lambda,n_projections,8)
    #     timeend=time.time()
    #     time2=timeend-timestart

    #     timestart=time.time()
    #     sopt(X,Y,Lambda,n_projections)
    #     timeend=time.time()
    #     time3=timeend-timestart
       

    #     time1_list.append(time1)
    #     time2_list.append(time2)
    #     time3_list.append(time3)
    # plt.plot(range(start_n,end_n,step),time1_list,label='sopt pool1')
    # plt.plot(range(start_n,end_n,step),time2_list,label='sopt pool2')
    # plt.plot(range(start_n,end_n,step),time3_list,label='sopt')
    
    # plt.xlabel("n: size of X")
    # plt.ylabel("runing time")
    # plt.legend(loc='best')
    # plt.show()



# from multiprocessing import Pool, TimeoutError
# import time
# import os

# def f(x):
#     return x*x

# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=4) as pool:

#         # print "[0, 1, 4,..., 81]"
#         print(pool.map(f, range(10)))

#         # print same numbers in arbitrary order
#         for i in pool.imap_unordered(f, range(10)):
#             print(i)

#         # evaluate "f(20)" asynchronously
#         res = pool.apply_async(f, (20,))      # runs in *only* one process
#         print(res.get(timeout=1))             # prints "400"

#         # evaluate "os.getpid()" asynchronously
#         res = pool.apply_async(os.getpid, ()) # runs in *only* one process
#         print(res.get(timeout=1))             # prints the PID of that process

#         # launching multiple evaluations asynchronously *may* use more processes
#         multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#         print([res.get(timeout=1) for res in multiple_results])

#         # make a single worker sleep for 10 secs
#         res = pool.apply_async(time.sleep, (10,))
#         try:
#             print(res.get(timeout=1))
#         except TimeoutError:
#             print("We lacked patience and got a multiprocessing.TimeoutError")

#         print("For the moment, the pool remains available for more work")

#     # exiting the 'with'-block has stopped the pool
#     print("Now the pool is closed and no longer available")
        

            

# d=X.shape[1]
# device=X.device.type
# dtype=X.dtype
# projections=random_projections(d,n_projections,device,dtype)
# X_sliced=torch.matmul(projections.T,X.T)
# Y_sliced=torch.matmul(projections.T,Y.T)
# cost_sum=0
# task_list=[{'X':X_sliced[i,:].detach(),'Y':Y_sliced[i,:].detach(),'Lambda':Lambda} for i in range(n_projections)]
# task=task_list[0]
# a=one_slice(task)
# X_take=X_sliced[0:].take(a[0])
# Y_take=Y_sliced[0:].take(a[1])                          
# torch.cat([X_take,X_take])
# #X_take=torch.cat([X_sliced[i:].take(plans[i][0]) for i in range(n_projections)])
# #Y_take=torch.cat([Y_sliced[i:].take(plans[i][1]) for i in range(n_projections)])
# total_cost=torch.sum(cost_function_T(X_take,Y_take))/n_projections