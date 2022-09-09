#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 20:16:47 2022

@author: baly
"""

n=10000
m=n+1000
Lambda=np.float32(100.0)
X=np.random.uniform(-20,20,n).astype(np.float32)
Y=np.random.uniform(-40,40,m).astype(np.float32)
X1=X.copy()
Y1=Y.copy()
start_time = time.time()
X1.sort()
Y1.sort()        
cost1,L1=opt_1d_v2(X1,Y1,Lambda)
end_time = time.time()
total_time=end_time-start_time
print(total_time)