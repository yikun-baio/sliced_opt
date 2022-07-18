# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:57:34 2022

@author: Yikun Bai
Yikun.bai@Vanderbilt.edu
"""

import numpy as np
import math
import torch
import os
#os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'

import numba as nb 
#from numba.types import Tuple
from typing import Tuple
import sys
from numba.typed import List
work_path=os.path.dirname(__file__)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)
os.chdir(parent_path)


from sopt2.library import *


def opt_1d_v1(X: np.array,Y: np.array,Lambda: np.float32) -> 'np.float,np.array':
    '''
    Parameters
    ------
    X: array_list shape, shape(n,) or (n,1), X should be shorted, float
    Y: array_list shape, shape(n,) or (n,1), Y should be shorted, float
    Lambda: float number >=0
    Returns:
    -----
    cost: float, cost of opt problem
    L: n*1 list, entry could be 0,1,2.... or -1
    where L[i]=j denote x_i->y_j and L[i]=-1 denotes we destroy x_i
    L must be in increasing order,
    if L=[0,1,1,2,-1], or L=[1,2,5,3], there is a bug. 
    optimal transportation plan
    
    '''
    def cost_plan_select():
        nonlocal L
        nonlocal cost
        cost_book={}    
        #select the best cost
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+(Lambda)  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=cost_function(xk,Y[j_last+1])
                cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2c': # cost for recursive problem
            #print('conflict')
                cost_xk_yjlast=cost_function(xk,Y[j_last])
                if cost_xk_yjlast<(Lambda):            
                    X1=X[0:k]
                    Y1=Y[0:j_last]
                    cost_sub,L_sub=opt_1d_v1(X1,Y1,Lambda)
                    cost_book[case]=cost_sub+cost_xk_yjlast
        # Update L 
 
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        #print(cost_book)
        
        if min_case=='1n':
            L.append(jk)
        elif min_case=='1c':
            L.append(j_last+1)
        elif min_case=='0':
            L.append(-1)
        elif min_case=='2c':
            L=L_sub+[j_last]
   
             
    n=len(X)
    m=len(Y)
    L=[]
    cost=0  
    if m==0:
        cost,L,xx,yy=empty_Y_opt(n,Lambda)
        return cost,L.tolist()
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2c'} 
        # 0 is for cost with destrouction, 1n is for cost without conflict, 1n,2n is the cost with conflict
        jk,cost_xk_yjk=closest_y(xk,Y)
        

        if cost_xk_yjk>=(Lambda): # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2c'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2c'}
            cost_plan_select()
            continue
        j_last=max(L) # index of last aligned y
        if jk>j_last:# No conflict
            case_set=case_set-{'1c','2c'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
    return cost,L

@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:],nb.float32,nb.int64[:]))(nb.float32[:,:],nb.int64[:],nb.float32)],nopython=True)
def opt_sub(M1,L1,Lambda):
    '''
    Parameters
    ------
    X1: array_list shape, shape(n,) or (n,1), X should be shorted, float 
    Y1: array_list shape, shape(n,) or (n,1), Y should be shorted, float
    L1: transportation plan of previous iteration, whose entries should be 0,1,2,...
    if L1 contains -1, there is a bug in the main loop. 
    Lambda: float number >=0
    
    
    Returns:
    -----
    cost_sub: float, cost of sub-opt problem
    L_sub: list, optimal transportation plan, whose entry must be 0,1,... or -1.
          L_sub can contains at most one -1, if it contains more than 1, there is a bug in this function.
    cost_sub_pre, float, cost of previous problem of the sub-opt problem
    L_sub_pre: list, optimal transprotation plan of previous problem, whose entry must be 0,1,...or -1
              L_sub_pre[end]=-1 or L_sub_pre=[], otherwise, there is bug in this function. 
    
    '''
    n1,m1=M1.shape
    L_sub_pre=np.empty(0,dtype=np.int64)
    cost_sub_pre=np.float32(0)
    cost_sub=np.float32(0)
    L_sub=np.empty(0,dtype=np.int64)
    
    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt(n1,Lambda)
        #print('sub funciton L sub',L_sub)
        #print('sub function n1',n1)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
    
    i_act,j_act=unassign_y(L1) # i_act is the index for L_sub, not the original_index


    L1=L1[0:n1-1]
    
    if L1.shape[0]==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt(M1,i_act,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre


    L1_inact=L1[0:i_act]
    L1x_inact=arange(0,i_act)
    cost_inact=np.float32(0)
    cost_inact=np.sum(matrix_take(M1,L1x_inact,L1_inact))
    
    if i_act==n1-1:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt(M1,i_act,j_act,Lambda)
        cost_sub=cost_inact+cost_sub      
        L_sub=np.concatenate((L1_inact,L_sub))

        if -1 ==L_sub[i_act]:
            cost_sub_pre=cost_sub 
            L_sub_pre=L_sub.copy()            
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
  

    # find the optimal d1 plan    
    L1_act=L1[i_act:]
    L1x_act=arange(i_act,n1)
    n_L1=n1-i_act
    

    cost_L1=matrix_take(M1,L1x_act[1:n_L1],L1_act) # cost list with 1 left shift
    cost_L2=matrix_take(M1,L1x_act[0:n_L1-1],L1_act) # cost list of original plan

    s=cost_L2.shape[0]        
    cost_list=np.concatenate((cost_L1,cost_L2))
    cost_d1=np.zeros(s+1,dtype=np.float32)
    cost_d1_opt=np.float32(np.inf)
    for i in range(s,-1,-1):
        cost_d1[i]=np.sum(cost_list[i:i+s])
        if i-1>=0 and cost_list[i-1]>=Lambda:
            cost_d1=cost_d1[i:]
            break
   
    index_d1_opt=cost_d1.argmin()+i
    cost_d1_opt=cost_d1.min()+Lambda
    #find the optimal d0 plan
    cost_d0=np.float32(np.inf)
    if j_act>=0 and i==0:
        cost_d0=cost_d1[0]+M1[i_act,j_act]
        
    if cost_d1_opt<=cost_d0:    
        cost_sub=cost_inact+cost_d1_opt
        L_sub=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64),L1_act[index_d1_opt:]))    
        
        cost_sub_pre=cost_inact+np.sum(cost_L2[0:index_d1_opt])+Lambda
        L_sub_pre=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64)))

          
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
    else:
        
        L_sub=np.concatenate((L1_inact,np.array([j_act],dtype=np.int64),L1_act))
        cost_sub=cost_inact+cost_d0
        cost_sub_pre=np.float32(0)
        L_sub_pre=np.empty(0,dtype=np.int64)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
        

@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:]))(nb.float32[:],nb.float32[:],nb.float32)],nopython=True)
def opt_1d_v2(X,Y,Lambda):
    M=cost_matrix(X,Y)
    n,m=M.shape
    Lambda=np.float32(Lambda)
    if m==0:
        cost,L,xx,yy=empty_Y_opt(n, Lambda)
        return cost,L
 
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    cost=np.float32(0) # save the optimal cost
    argmin_Y=closest_y_M(M)

    # initial the subproblem
    cost_pre=np.float32(0)
    L_pre=np.empty(0,dtype=np.int64)
    cost_sub_pre=np.float32(0)
    L_sub_pre=np.empty(0,dtype=np.int64)
    i_start=np.int64(0)
    j_start=np.int64(0)
    

    
    # initial loop    
    k=0
    jk=argmin_Y[k]
    cost_xk_yjk=M[k,jk]    
    if cost_xk_yjk<Lambda:
        cost+=cost_xk_yjk
        L=np.concatenate((L,np.array([jk],dtype=np.int64)))    
    else: #  we destroy a point 
        cost+=Lambda
        L=np.concatenate((L,np.array([-1],dtype=np.int64))) 
        cost_pre=cost
        L_pre=L.copy()
        i_start,j_start=startindex(L_pre)
     
    # start loop 
    # 0 denote the case destroy point, 1 denote the case assign to y_jk, 2 denote the case assign to y_jlast+1, 3 denot the case assign to y_jlast 
    cost_book_orig=np.array([np.inf,np.inf,np.inf,np.inf],dtype=np.float32)

    for k in range(1,n):
        cost_book=cost_book_orig.copy()
        if j_start==m: # There is no y, so we destroy point
            cost_end,L_end,xx,yy=empty_Y_opt(n-k, Lambda)
            cost=cost+cost_end
            L=np.concatenate((L,L_end))
            return cost,L
        
        jk=argmin_Y[k]
        cost_xk_yjk=M[k,jk]
        
        j_last=L[-1] # index of last y 
        if j_last<0:
            j_last=j_start-1
            
        # case of no conflict
        if jk>j_last:# No conflict L[-1]=j_last
            cost_book[0]=cost+Lambda
            cost_book[1]=cost+cost_xk_yjk
        
        else:# conflict
            cost_book[0]=cost+Lambda
    
            # cost 1c
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost_book[2]=cost+cost_xk_yjlast1
            #cost2c
            cost_xk_yjlast=M[k,j_last]                  
            if cost_xk_yjlast<Lambda and j_start<=m and i_start<k:
                M1=M[i_start:k,j_start:j_last]
                X1=X[i_start:k]
                Y1=Y[j_start:j_last]
                L1=L[i_start:k].copy()

        #     # we need the last assign index since we need to retrieve the closest unassigend j                    
                
                index_adjust(L1,-j_start)
                cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub(M1,L1,Lambda)
                cost_book[3]=cost_pre+cost_sub+cost_xk_yjlast                       
                index_adjust(L_sub,j_start)
                index_adjust(L_sub_pre,j_start)


        # # find the optimal cost over all 
        min_case=cost_book.argmin()
        cost=cost_book.min()
        
        # update problem, if we destroy points, update pre_problem
        if min_case==0:
            L=np.concatenate((L,np.array([-1],dtype=np.int64)))
            # update previous problem 
            cost_pre=cost
            L_pre=L.copy()
            i_start,j_start=startindex(L_pre)
        elif min_case==1:
            L=np.concatenate((L,np.array([jk],dtype=np.int64)))
        elif min_case==2:
            L=np.concatenate((L,np.array([j_last+1],dtype=np.int64)))
        elif min_case==3:
            L=np.concatenate((L_pre,L_sub,np.array([j_last],dtype=np.int64)))
            # update previous problem 
            if L_sub_pre.shape[0]>=1:
                cost_pre=cost_pre+cost_sub_pre
                L_pre=np.concatenate((L_pre,L_sub_pre))      
            # empty the variable for sub problem
            L_sub=np.empty(0,dtype=np.int64)
            L_sub_pre=np.empty(0,dtype=np.int64)
            cost_sub=np.float32(0)
            cost_sub_pre=np.float32(0)
            i_start,j_start=startindex(L_pre)
        
      
    return cost,L


@nb.njit()
def opt_decomposition(X,Y,Lambda):
    M=cost_matrix(X,Y)
    CM=M>=Lambda
    n,m=M.shape
    argmin_Y=closest_y_M(M)
    mass_loss=0
    
    Occupy=np.zeros(m,dtype=np.int64)
    free_Y=np.empty(0,dtype=np.int64)
    X_list=List()
    Y_list=List()

    k=0
    jk=argmin_Y[k]
#    cxy=M[k,jk]
    previous_index=-1
    if CM[k,jk]==1:
    
        X_list.append(np.array([k]))            
        Y_list.append(np.empty(0,dtype=np.int64))
        free_Y=np.concatenate((free_Y,np.array([-1])))
        previous_index=len(X_list)-1
        mass_loss+=1
    else:
        X_list.append(np.array([k]))            
        Y_list.append(np.array([jk]))
        free_y=jk-1
        if free_y>=0 and Occupy[free_y]==1:
            free_y=free_Y[-1]
        free_Y=np.concatenate((free_Y,np.array([free_y])))
        Occupy[jk]=1
    
    for k in range(1,n):
        
        jk=argmin_Y[k]
        if Occupy[jk]==0:
            #
            if CM[k,jk]==1:
                X_list.append(np.array([k]))            
                Y_list.append(np.empty(0,dtype=np.int64))
                free_Y=np.concatenate((free_Y,np.array([-1])))
                previous_index=len(X_list)-1
            else:
                X_list.append(np.array([k]))            
                Y_list.append(np.array([jk]))
                free_y=jk-1
                if Occupy[free_y]==1:
                    free_y=free_Y[-1]
                free_Y=np.concatenate((free_Y,np.array([free_y])))
                Occupy[jk]=1     

        elif Occupy[jk]==1: # If it is occupied, we should extend the last problem
             
            #check if we need to merge subproblems
            last_free_y=free_Y[-1] # it must be occupied by last problem 
            
            if last_free_y==-1:
                index_start=previous_index
            else:                  
                last_index=free_Y.shape[0]-1
                index_start=np.where(free_Y==last_free_y)[0][0]
            
            if index_start<last_index:#Then we need to merge all the previous problems 
                merged_X=X_list[last_index]
                merged_Y=Y_list[last_index]
                for i in range(last_index-1,index_start-1,-1):
                    Y_take=Y_list[i]
                    y_index=Y_take[-1]
                    X_take=X_list[i+1]
                    x_index=X_take[0]
                    if CM[x_index,y_index]:
                        previous_index=i
                        break
                    merged_X=np.concatenate((X_list[i],merged_X))
                    merged_Y=np.concatenate((Y_list[i],merged_Y))
                    
                
#                merged_X=merge_list(X_list[index_start:last_index+1])
#                merged_Y=merge_list(Y_list[index_start:last_index+1])
                X_list=X_list[0:i]
                X_list.append(merged_X)
                Y_list=Y_list[0:i]
                Y_list.append(merged_Y)
                free_Y=free_Y[0:i+1]
                if i>index_start:
                    free_Y[-1]=-1


            #extend the last problem      
            last_X=X_list[-1] # the last problem must have jk
            last_Y=Y_list[-1]
            last_y=last_Y[-1]
            last_free_y=free_Y[-1]
            last_X=np.concatenate((last_X,np.array([k])))    
            #extend last_X
            X_list=X_list[0:-1]
            X_list.append(last_X)
            # estend last_Y from left 
            x_start=last_X[0]
            if last_free_y>=0 and CM[x_start,last_free_y]==0:
                last_Y=np.concatenate((np.array([last_free_y]),last_Y))
                Occupy[last_free_y]=1
            
            #extend last_Y from right
            x_end=last_X[-1]
            if last_y+1<=m-1 and CM[x_end,last_y+1]==0:
                last_Y=np.concatenate((last_Y,np.array([last_y+1])))
                Occupy[last_y+1]=1
            Y_list=Y_list[0:-1]
            Y_list.append(last_Y)
                
            free_y=last_free_y-1
            if free_y>=0 and Occupy[free_y]==1:
                free_y=free_Y[-2]
            free_Y[-1]=free_y
    return X_list,Y_list,free_Y

@nb.njit([nb.types.Tuple((nb.float32,nb.int64[:]))(nb.float32[:],nb.float32[:],nb.float32)]) #,parallel=True)
def opt_1d_v3(X,Y,Lambda):
    X_list,Y_list,free_Y=opt_decomposition(X,Y,Lambda)
    K=len(X_list)
    costs=np.zeros(K,dtype=np.float32)
    plans=List()
#    plans.append(np.array([0]))
    for i in range(K):
#        n=X_list[i].shape[0]
        plans.append(np.empty(0,dtype=np.int64))  
    
    for i in nb.prange(K):
        indices_X=X_list[i]
        indices_Y=Y_list[i]
        Xs=X[indices_X]
        Ys=Y[indices_Y]
        cost,L=opt_1d_v2(Xs,Ys,Lambda)
        
        if indices_Y.shape[0]>0:
            index_adjust(L,indices_Y[0])       
        plans[i]=np.concatenate((plans[i],L))
        costs[i]=cost
    L=merge_list(plans)
    cost=np.sum(costs)
    return cost,L

@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:],nb.float32,nb.int64[:]))(nb.float32[:,:],nb.int64[:],nb.float32)],nopython=True)
def opt_sub_np(M1:nb.float32[:,:],L1:nb.int64[:],Lambda:nb.float32):    
    '''
    Parameters
    ------
    M1: n*m np matrix, float 
    L1: transportation plan of previous iteration, whose entries should be 0,1,2,...
        if L1 contains -1, there is a bug in the main loop. 
    Lambda: float number >=0
    
    
    Returns:
    -----
    cost_sub: float, cost of sub-opt problem
    L_sub: list, optimal transportation plan, whose entry must be 0,1,... or -1.
          L_sub can contains at most one -1, if it contains more than 1, there is a bug in this function.
    cost_sub_pre, float, cost of previous problem of the sub-opt problem
    L_sub_pre: list, optimal transprotation plan of previous problem, whose entry must be 0,1,...or -1
             L_sub_pre[end]=-1 or L_sub_pre=[], otherwise, there is bug in this function. 
    
    '''
    
    n1,m1=M1.shape
    L_sub_pre=np.empty(0,dtype=np.int64)
    cost_sub_pre=np.float32(0)
    cost_sub=np.float32(0)
    L_sub=np.empty(0,dtype=np.int64)
    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt(n1,Lambda)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
    i_act,j_act=unassign_y(L1) # i_act is the index for L_sub, not the original_index
    L1=L1[0:n1-1]
    
    if L1.shape[0]==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_np(M1,i_act,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre

    L1_inact=L1[0:i_act]
    L1x_inact=arange(0,i_act)  
    
    cost_inact=np.sum(matrix_take(M1,L1x_inact,L1_inact))

    if i_act==n1-1:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_np(M1,i_act,j_act,Lambda)
        cost_sub=cost_inact+cost_sub      
        L_sub=np.concatenate((L1_inact,L_sub))
        if -1 ==L_sub[i_act]:
            cost_sub_pre=cost_sub 
            L_sub_pre=L_sub.copy()            
            return cost_sub,L_sub,cost_sub_pre,L_sub_pre

    # find the optimal d1 plan    
    L1_act=L1[i_act:]
    L1x_act=arange(i_act,n1)
   


    cost_L1=matrix_take(M1,L1x_act[1:],L1_act) # cost of 1 shift plan 
    cost_L2=matrix_take(M1,L1x_act[0:-1],L1_act) # cost of original shift 
    s=cost_L2.shape[0]
    cost_list=np.concatenate((cost_L1,cost_L2))
    C=np.zeros((2*s,s+1),dtype=np.float32)
    ones=np.ones(s,dtype=np.float32)
    for i in range(s+1):
        C[i:i+s,i]=ones
 
    cost_d1=np.dot(cost_list,C)
    index_d1_opt=cost_d1.argmin()
    cost_d1_opt=cost_d1.min()+Lambda
  
    #find the optimal d0 plan
    cost_d0=np.float32(np.inf)
    if j_act>=0:
        cost_d0=cost_d1[0]+M1[i_act,j_act]
        
    if cost_d1_opt<=cost_d0:
        cost_sub=cost_inact+cost_d1_opt
        L_sub=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64),L1_act[index_d1_opt:]))     
        cost_sub_pre=cost_inact+np.sum(cost_L2[0:index_d1_opt])+Lambda
        L_sub_pre=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64)))         
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
        
    else:
        L_sub=np.concatenate((L1_inact,np.array([j_act],dtype=np.int64),L1_act))
        cost_sub=cost_inact+cost_d0
        cost_sub_pre=np.float32(0)
        L_sub_pre=np.empty(0,dtype=np.int64)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre


@nb.jit([nb.types.Tuple((nb.float32,nb.int64[:]))(nb.float32[:],nb.float32[:],nb.float32)],nopython=True)
def opt_1d_np(X,Y,Lambda):
    '''
    Parameters
    ------
    X: array_list shape, shape(n,) or (n,1), X should be shorted, float 
    Y: array_list shape, shape(n,) or (n,1), Y should be shorted, float
    Lambda: float number >=0
    
    Returns:
    -----
    cost: float, cost of opt problem
    L: n*1 list, entry could be 0,1,2.... or -1
    where L[i]=j denote x_i->y_j and L[i]=-1 denotes we destroy x_i
    L must be in increasing order,
    if L=[0,1,1,2,-1], or L=[1,2,5,3], there is a bug. 
    optimal transportation plan
    
    '''
   
    M=cost_matrix(X,Y)

    n,m=M.shape
    Lambda=np.float32(Lambda)
 
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    cost=np.float32(0) # save the optimal cost
    
    argmin_Y=closest_y_M(M)

    # initial the sub
    cost_pre=np.float32(0)
    L_pre=np.empty(0,dtype=np.int64)
    cost_sub_pre=np.float32(0)
    L_sub_pre=np.empty(0,dtype=np.int64)
    i_start=0
    j_start=0
    

    
    # initial loop    
    k=0
    jk=argmin_Y[k]
    cost_xk_yjk=M[k,jk]    
    if cost_xk_yjk<Lambda:
        cost+=cost_xk_yjk
        L=np.concatenate((L,np.array([jk],dtype=np.int64)))    
    else: #  we destroy a point 
        cost+=Lambda
        L=np.concatenate((L,np.array([-1],dtype=np.int64))) 
        cost_pre=cost
        L_pre=L.copy()
        i_start,j_start=startindex(L_pre)
     
    # start loop 
    # 0 denote the case destroy point, 1 denote 
    cost_book_orig=np.array([np.inf,np.inf,np.inf,np.inf],dtype=np.float32)

    for k in range(1,n):
        cost_book=cost_book_orig.copy()
        if j_start==m: # There is no y, so we destroy point
            cost_end,L_end,xx,yy=empty_Y_opt(n-k, Lambda)
            cost=cost+cost_end
            L=np.concatenate((L,L_end))
            return cost,L
        
        jk=argmin_Y[k]
        cost_xk_yjk=M[k,jk]
        
        j_last=L[-1] # index of last y 
        if j_last<0:
            j_last=j_start-1
            
        # case of no conflict
        if jk>j_last:# No conflict L[-1]=j_last
            cost_book[0]=cost+Lambda
            cost_book[1]=cost+cost_xk_yjk
        
        else:# conflict
            cost_book[0]=cost+Lambda
    
            # cost 1c
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost_book[2]=cost+cost_xk_yjlast1
            #cost2c
            cost_xk_yjlast=M[k,j_last]                  
            if cost_xk_yjlast<Lambda and j_start<=m and i_start<k:
                M1=M[i_start:k,j_start:j_last]
                L1=L[i_start:k].copy() 
        #     # we need the last assign index since we need to retrieve the closest unassigend j                    
                index_adjust(L1,-j_start)
                cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub_np(M1,L1,Lambda)
                cost_book[3]=cost_pre+cost_sub+cost_xk_yjlast                       
                index_adjust(L_sub,j_start)
                index_adjust(L_sub_pre,j_start)
        # # find the optimal cost over all 
        min_case=cost_book.argmin()
        cost=cost_book.min()
        
        # update problem, if destroy points, update pre_problem)
  
        if min_case==0:
            L=np.concatenate((L,np.array([-1],dtype=np.int64)))
            # update previous problem 
            cost_pre=cost
            L_pre=L.copy()
            i_start,j_start=startindex(L_pre)
        elif min_case==1:
            L=np.concatenate((L,np.array([jk],dtype=np.int64)))
        elif min_case==2:
            L=np.concatenate((L,np.array([j_last+1],dtype=np.int64)))
        elif min_case==3:
            L=np.concatenate((L_pre,L_sub,np.array([j_last],dtype=np.int64)))
            # update previous problem 
            if L_sub_pre.shape[0]>=1:
                cost_pre=cost_pre+cost_sub_pre
                L_pre=np.concatenate((L_pre,L_sub_pre))
                
                # empty the variable for sub problem
            L_sub=np.empty(0,dtype=np.int64)
            L_sub_pre=np.empty(0,dtype=np.int64)
            cost_sub=np.float32(0)
            cost_sub_pre=np.float32(0)
            i_start,j_start=startindex(L_pre)




    return cost,L

        
@torch.jit.script 
def opt_sub_T(M1,L1,Lambda):
    device=M1.device.type
    n1,m1=M1.shape
    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt_T(n1,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre
    i_act,j_act=unassign_y_T(L1) # i_act is the index for L_sub, not the original_index

    L1=L1[:-1]
    if L1.shape[0]==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_T(M1,i_act,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre   
    
    # divide the problem 
    L1_inact=L1[0:i_act] 
    Lx1_inact=torch.arange(0,i_act)
    cost_inact=torch.sum(M1[Lx1_inact,L1_inact])
    L1_act=L1[i_act:]
    if i_act==n1-1:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_T(M1,i_act,j_act,Lambda)
        cost_sub=cost_inact+cost_sub      
        L_sub=torch.cat([L1_inact,L_sub])
        if L_sub[i_act]==-1:
            cost_sub_pre=cost_sub 
            L_sub_pre=L_sub.clone()

        return cost_sub,L_sub,cost_sub_pre,L_sub_pre

    # find the optimal d1 plan
    L1_act=L1[i_act:]
    L1x_act=torch.arange(i_act,n1) 
    cost_L1=M1[L1x_act[1:],L1_act]
    cost_L2=M1[L1x_act[0:-1],L1_act]
    s=cost_L2.shape[0]
    cost_list=torch.cat([cost_L1,cost_L2])
    #C=torch.zeros([2*s,s+1],device=device)
    ones=torch.ones(s,device=device)
    C=torch.stack([torch.cat([torch.zeros(i,device=device),ones,torch.zeros(s-i,device=device)]) for i in range(s+1)]).T
#    for i in range(s+1):
#        C[i:i+s,i]=ones
    cost_d1=torch.matmul(cost_list,C)
    cost_d1_opt=cost_d1.min()+Lambda
    index_d1_opt=cost_d1.argmin()
    
    # find the optimal d0 plan
    cost_d0=torch.inf    
    if j_act>=0:
        cost_d0=cost_d1[0]+M1[i_act,j_act]

    if cost_d1_opt<=cost_d0:
        cost_sub=cost_inact+cost_d1_opt
        L_sub=torch.cat([L1_inact,L1_act[0:index_d1_opt],torch.tensor([-1],device=device),L1_act[index_d1_opt:]])     
        cost_sub_pre=cost_inact+torch.sum(cost_L2[0:index_d1_opt])+Lambda
        L_sub_pre=torch.cat([L1_inact,L1_act[0:index_d1_opt],torch.tensor([-1],device=device)])         

        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
        
    else:
        L_sub=torch.cat([L1_inact,j_act.reshape(1),L1_act])
        cost_sub=cost_inact+cost_d0
        cost_sub_pre=torch.tensor(0.0,device=device)
        L_sub_pre=torch.empty(0,device=device,dtype=torch.int64)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre

    
@torch.jit.script 
def opt_1d_T(X,Y,Lambda: float):
    '''
    Parameters
    ----------
    X : n*1 sorted tensor
        value of one distribution 

    Y : m*1 sorted tensor
        value of one distirbution 
    Lambda : float number
        control the mass penulty 

    Returns
    -------
    cost: cost of opt (X,Y;lambda)
    L: transportation plan

    '''

    
    device=X.device.type   
    M=cost_matrix_T(X,Y)
    min_Y=M.argmin(dim=1)
    n,m=M.shape

    Lambda=torch.tensor(Lambda,device=device,dtype=torch.float32)
    
    L=torch.empty(0,dtype=torch.int64,device=device) # save the optimal plan
    cost=torch.tensor(0.0,device=device) # save the optimal cost
    cost_pre=torch.tensor(0.0,device=device)
    L_pre=torch.empty(0,dtype=torch.int64,device=device)
    cost_sub_pre=torch.empty(0,device=device)
    L_sub_pre=torch.empty(0,device=device)
    i_start=0
    j_start=0


     # subproblem 
    L_sub=torch.empty(0,device=device,dtype=torch.float32)
    L_sub_pre=torch.empty(0,device=device,dtype=torch.float32)
    cost_sub=torch.tensor(0,device=device,dtype=torch.float32)
    cost_sub_pre=torch.tensor(0,device=device,dtype=torch.float32)
    
#    initial loop
    k=0
    jk=min_Y[k]
    cost_xk_yjk=M[k,jk]
#    case_set_orig=torch.tensor([0,1,2,3],dtype=torch.int64)
    cost_book_orig=torch.tensor([torch.inf,torch.inf,torch.inf,torch.inf],dtype=torch.float32)
    
    if cost_xk_yjk<Lambda:
        cost+=cost_xk_yjk
        L=torch.cat([L,jk.reshape(1)])    
    else: #  we destroy a point 
        cost+=Lambda
        L=torch.cat([L,torch.tensor([-1],device=device)]) 
        cost_pre=cost
        L_pre=L.clone()
        i_start,j_start=startindex_T(L_pre)
        


    for k in range (1,n):
#   case_set # 0 is for cost with destroying, 1n is for cost without conflict, 1c,2c are cost with conflict

        
        if j_start==m: # There is no y, so we destroy point
            cost_end,L_end,xx,yy=empty_Y_opt_T(n-k, Lambda)
            cost=cost+cost_end
            L=torch.cat((L,L_end))
            return cost,L

        
        jk=min_Y[k]
        cost_xk_yjk=M[k,jk]
        cost_book=cost_book_orig.clone()

        
        j_last=L[k-1].clone() #   index of last y
#        cost_sub,L_sub,cost_sub_pre,L_sub_pre

        if j_last<0:
            j_last=torch.tensor(j_start-1,device=device)

        if jk>j_last:# No conflict L[-1]=j_last do it in normal case
            cost_book[0]=cost+Lambda
            cost_book[1]=cost+cost_xk_yjk
           
        else:# there is conflict
            cost_book[0]=cost+Lambda
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost_book[2]=cost+cost_xk_yjlast1
        
            cost_xk_yjlast=M[k,j_last]                
            if cost_xk_yjlast<Lambda and j_start<=m and i_start<k:
                L1=L[i_start:k].clone() # we need the last assign index since we need to retrieve the closest unassigned y
                index_adjust_T(L1,-j_start)
                M1=M[i_start:k,j_start:j_last]
                cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub_T(M1,L1,Lambda)
                cost_book[3]=cost_pre+cost_sub+cost_xk_yjlast   
                index_adjust_T(L_sub,j_start)
                index_adjust_T(L_sub_pre,j_start)


        # find the optimal cost over all 
        min_case=cost_book.argmin()
        cost=cost_book.min()
    
          # update problem, if destroy points, update pre_problem)
  
        if min_case==0:
            L=torch.cat([L,torch.tensor([-1],device=device)])
            # update previous problem 
            cost_pre=cost
            L_pre=L.clone()
            i_start,j_start=startindex_T(L_pre)
        elif min_case==1:
            L=torch.cat([L,jk.reshape(1)])
        elif min_case==2:
            L=torch.cat([L,(j_last+1).reshape(1)])
        elif min_case==3:
            L=torch.cat([L_pre,L_sub,j_last.reshape(1)])
            # update previous problem 
            if L_sub_pre.shape[0]>=1:
                cost_pre=cost_pre+cost_sub_pre
                L_pre=torch.cat((L_pre,L_sub_pre))
            L_sub=torch.empty(0,device=device,dtype=torch.int64)
            L_sub_pre=torch.empty(0,device=device,dtype=torch.int64)
            cost_sub=torch.tensor(0,device=device,dtype=torch.float32)
            cost_sub_pre=torch.tensor(0,device=device,dtype=torch.float32)
            i_start,j_start=startindex_T(L_pre)

        
    return cost,L


@nb.jit(nopython=True)
def pot_1d(X,Y): 
    n=len(X)
    m=len(Y)
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    cost=0.0 # save the optimal cost
#    M=cost_matrix_T(X,Y)
  #   min_Y=M.argmin(dim=1)
  #   n,m=M.shape
 
    #initial loop:
    k=0
    xk=X[k]

    jk,cost_xk_yjk=closest_y(xk,Y)
    cost+=cost_xk_yjk
    L=np.append(L,jk)

    for k in range(1,n):
        xk=X[k]
        jk,cost_xk_yjk=closest_y(xk,Y)
        j_last=L[-1]
        
        #define consistent term 

        
        
        if jk>j_last:# No conflict, L[-1] is the j last assig
            jk,cost_xk_yjk=closest_y(xk,Y)
            cost+=cost_xk_yjk
            L=np.append(L,jk)
        else:
            # this is the case for conflict: 

            # compute the first cost 
            if j_last+1<=m-1:
                ylast1=Y[j_last+1]
                c_xk_yjlast1=cost_function(xk,ylast1)
                cost1=cost+c_xk_yjlast1
            else:
                cost1=np.inf 
            
            # compute the second cost 
            i_act,j_act=unassign_y(L)
            if j_act>=0:                        
                L1=np.concatenate((L[0:i_act],np.array([j_act]),L[i_act:]))
                Y_assign=Y[L1]
                X_assign=X[0:k+1]
                cost2=np.sum(cost_function(X_assign,Y_assign))
            else:
                cost2=np.inf
            if cost1<cost2:
                cost=cost1
                L=np.append(L,j_last+1)
    
            elif cost2<=cost1:
                cost=cost2
                L=L1.copy()    
    return cost,L






n=200
m=n+300
#Lambda=4
Lambda=np.float32(0.01)
for i in range(10):
    X=torch.rand(n,dtype=torch.float32)*4-0.5
    Y=torch.rand(m,dtype=torch.float32)*3+0.8
    X=X.sort().values
    Y=Y.sort().values
    X1=X.numpy()
    Y1=Y.numpy()
    cost1,L1=opt_1d_v3(X1,Y1,Lambda)
    cost2,L2=opt_1d_v2(X1,Y1,Lambda)
  #   cost3,L3=pot_1d(X1,Y1)
    if abs(cost1-cost2)>=0.0001:
        print('error')

# X=np.array([0.47016394, 1.0328217 , 1.3818892 , 1.4757019 , 1.9214936 ,
#        2.1895437 , 2.604464  , 2.7063289 , 2.7503216 , 2.922912  ],
#       dtype=np.float32)

# Y=np.array([-0.93786955, -0.8235371 , -0.658347  , -0.4214544 , -0.3613819 ,
#        -0.11025443, -0.06072737,  0.02043935,  0.11602888,  0.22903188,
#         0.4837798 ,  0.5873061 ,  0.7503131 ], dtype=np.float32)
# X1=torch.from_numpy(X)
# Y1=torch.from_numpy(Y)
# #X1=np.array([0.9788, 2.0892, 2.5528, 2.6119, 3.0876], dtype=np.float32)
# #Y1=np.array([4.1054, 4.5741, 4.6948, 5.3001, 5.4889, 5.7816, 6.1562, 6.3901],dtype=np.float32)
# cost2,L2=opt_1d_v2(X,Y,Lambda)
# cost1,L1=opt_1d_v1(X,Y,Lambda)
# opt_1d_np(X,Y,Lambda)
# X1=np.array([ 6.199362,  10.1927185])


# a,b,c,d=opt_sub_np(X1,Y1,L1,Lambda)

