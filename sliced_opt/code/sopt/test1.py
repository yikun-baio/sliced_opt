# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import torch.multiprocessing as mp
import torch
import time


# import sopt.library


try:
    mp.set_start_method('spawn', force=True)
    print("spawned")
except RuntimeError:
    pass


class test():
    def __init__(self,X,Y):
        self.X=X #.share_memory_()
        self.Y=Y
        self.n=X.shape[0]
        self.device=X.device.type
        self.results=torch.zeros([self.n]).to(self.device)
        self.run()
        
    def run(self):
        pool = mp.Pool(2)
        #for i in range(self.n):
        r=pool.map_async(self.f, range(self.n))
        pool.close()
        print('here')
        pool.join()
        print('result',self.results)
        
    def f(self,i):
        x=self.X[i]
        self.results[i]=2*x
        
#        self.result[i]=2*xi



    
    
    
