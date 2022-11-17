# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:57:34 2022

"""

import numpy as np
import math
import torch
import os

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
from sopt.library import *


@nb.njit(['Tuple((float64,int64[:],float64,int64[:]))(float64[:,:],int64[:],float64)'],fastmath=True,cache=True)
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
    dtype=type(Lambda)
    n1,m1=M1.shape
    L_sub_pre=np.empty(0,np.int64)
    cost_sub_pre=np.float64(0)
    cost_sub=np.float64(0)
    L_sub=np.empty(0,np.int64)
    
    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt(n1,Lambda)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre


    i_act,j_act=unassign_y(L1) # i_act is the index for L_sub, not the original_index
    L1=L1[0:n1-1]
    
    if L1.shape[0]==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt(M1,i_act,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre

    L1_inact=L1[0:i_act]
    L1x_inact=arange(0,i_act)
    cost_inact=np.float64(0)
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
    i=0
    
    i_list=np.where(cost_L1>=Lambda)[0]
    if i_list.shape[0]>=1:
        i_start=i_list[-1]+1
    else:
        i_start=0
    cost_d1=np.zeros(s+1-i_start,np.float64)

    
    for i in range(i_start,s+1):
        cost_d1[i-i_start]=np.sum(cost_list[i:i+s])

    #print(cost_d1)
    index_d1_opt=cost_d1.argmin()+i_start
    cost_d1_opt=cost_d1.min()+Lambda
    #find the optimal d0 plan
    cost_d0=np.float64(np.inf)
    if j_act>=0 and i_start==0:
        cost_d0=cost_d1[0]+M1[i_act,j_act]
        
    if cost_d1_opt<=cost_d0:    
        cost_sub=cost_inact+cost_d1_opt
        L_sub=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64),L1_act[index_d1_opt:]))    
        cost_sub_pre=cost_inact+np.sum(cost_L2[0:index_d1_opt])+Lambda
        L_sub_pre=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64)))
#        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
    else:
        L_sub=np.concatenate((L1_inact,np.array([j_act],dtype=np.int64),L1_act))
        cost_sub=cost_inact+cost_d0
        cost_sub_pre=np.float64(0)
        L_sub_pre=np.empty(0,np.int64)
#        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    return cost_sub,L_sub,cost_sub_pre,L_sub_pre
        

@nb.njit(['Tuple((float64,int64[:],float64,int64[:]))(float64[:,:],int64[:],float64)'],cache=True)
#@nb.njit()
def opt_sub_a(M1,L1,Lambda):
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
    dtype=type(Lambda)
    L_sub_pre=np.empty(0,np.int64)
    cost_sub_pre=np.float64(0)
    cost_sub=np.float64(0)
    L_sub=np.empty(0,np.int64)
    
    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt(n1,Lambda)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
    i_act,j_act=unassign_y(L1) # i_act is the index for L_sub, not the original_index
    L1=L1[0:n1-1]
    
    if L1.shape[0]==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt(M1,i_act,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre

    L1_inact=L1[0:i_act]
    L1x_inact=arange(0,i_act)
    cost_inact=np.float64(0)
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
    cost_L=np.concatenate((cost_L1,cost_L2))
    
    
    i_list=np.where(cost_L1>=Lambda)[0]
    if i_list.shape[0]>=1:
        i_start=i_list[-1]+1
    else:
        i_start=0
        
    cost_d11=np.sum(cost_L2)+Lambda 
    cost_d12=np.sum(cost_L[i_start:i_start+s])+Lambda
    i_rand=np.random.randint(i_start,s+1)
    cost_d13=np.sum(cost_L[i_rand:i_rand+s])+Lambda
    cost_list=np.array([cost_d11,cost_d12,cost_d13])
    min_index=cost_list.argmin()
    cost_d1_opt=cost_list.min()
    if min_index==0:
        index_d1_opt=s
        
    elif min_index==1:
        index_d1_opt=i_start
    elif min_index==2:
        index_d1_opt=i_rand

    cost_d0=np.float64(np.inf)
    if j_act>=0 and i_start==0:
        cost_d0=cost_d12-Lambda+M1[i_act,j_act]
        
    if cost_d1_opt<=cost_d0:    
        cost_sub=cost_inact+cost_d1_opt
        L_sub=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64),L1_act[index_d1_opt:]))    
        cost_sub_pre=cost_inact+np.sum(cost_L2[0:index_d1_opt])+Lambda
        L_sub_pre=np.concatenate((L1_inact,L1_act[0:index_d1_opt],np.array([-1],dtype=np.int64)))
#        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
    else:
        L_sub=np.concatenate((L1_inact,np.array([j_act],dtype=np.int64),L1_act))
        cost_sub=cost_inact+cost_d0
        cost_sub_pre=np.float64(0)
        L_sub_pre=np.empty(0,dtype=np.int64)
#        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    return cost_sub,L_sub,cost_sub_pre,L_sub_pre


@nb.njit(['Tuple((float64,int64[:]))(float64[:],float64[:],float64)'],cache=True)
def opt_1d_v2(X,Y,Lambda):
    dtype=type(Lambda)
    M=cost_matrix(X,Y)
    n,m=M.shape
    if m==0:
        cost,L,xx,yy=empty_Y_opt(n, Lambda)
        return cost,L
 
    L=np.empty(0,np.int64) # save the optimal plan
    cost=np.float64(0) # save the optimal cost
    argmin_Y=closest_y_M(M)

    # initial the subproblem
    cost_pre=np.float64(0)
    L_pre=np.empty(0,np.int64)
    cost_sub_pre=np.float64(0)
    L_sub_pre=np.empty(0,np.int64)
    i_start=0
    j_start=0
    
    # initial loop    
    k=0
    jk=argmin_Y[k]
    cost_xk_yjk=M[k,jk]    
    if cost_xk_yjk<Lambda:
        cost+=cost_xk_yjk
        L=np.concatenate((L,np.array([jk],np.int64)))    
    else: #  we destroy a point 
        cost+=Lambda
        L=np.concatenate((L,np.array([-1],np.int64))) 
        cost_pre=cost
        L_pre=L.copy()
        i_start=1
        j_start=0
     
    # start loop 
    # 0 denote the case destroy point, 1 denote the case assign to y_jk, 2 denote the case assign to y_jlast+1, 3 denot the case assign to y_jlast 
    cost_book_orig=np.full(4,np.inf,dtype=np.float64)
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
            # cost 2
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost_book[2]=cost+cost_xk_yjlast1
            #cost 3
            cost_xk_yjlast=M[k,j_last]                  
            if cost_xk_yjlast<Lambda and j_start<=m and i_start<k:
                M1=M[i_start:k,j_start:j_last]
                L1=L[i_start:k].copy()

        #     # we need the last assign index since we need to retrieve the closest unassigend j                    
                index_adjust(L1,-j_start)
                cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub(M1,L1,Lambda)
                cost_book[3]=cost_pre+cost_sub+cost_xk_yjlast                       
                index_adjust(L_sub,j_start)
                index_adjust(L_sub_pre,j_start)

        # find the optimal cost over all 
        #cost_book1=cost_book[np.array([3,0,1,2])]
        #min_case1=cost_book1.argmin()
        #cost=cost_book1.min()
        #min_case=np.array([3,0,1,2])[min_case1]
        min_case=cost_book.argmin()
        cost=cost_book.min()
        
        # update problem, if we destroy points, update pre_problem
        if min_case==0:
            L=np.append(L,-1)
            # update previous problem 
            cost_pre=cost
            L_pre=L.copy()
            j_start=j_last+1
            i_start=k+1


        elif min_case==1:
            L=np.append(L,jk) #],dtype=np.int64)))
        elif min_case==2:
            L=np.append(L,j_last+1) #],dtype=np.int64)))
        elif min_case==3:
            L=np.concatenate((L_pre,L_sub,np.array([j_last],dtype=np.int64)))
            # update previous problem 
            if L_sub_pre.shape[0]>=1:
                cost_pre=cost_pre+cost_sub_pre
                L_pre=np.concatenate((L_pre,L_sub_pre))
                i_start=i_start+L_sub_pre.shape[0]
                if L_sub_pre.shape[0]>=2:
                    j_start=L_sub_pre[-2]+1
            
            # empty the variable for sub problem
            L_sub=np.empty(0,dtype=np.int64)
            L_sub_pre=np.empty(0,dtype=np.int64)
            cost_sub=np.float64(0)
            cost_sub_pre=np.float64(0)
    return cost,L

@nb.njit(['Tuple((float64,int64[:]))(float64[:],float64[:],float64)'])
def opt_1d_v2_a(X,Y,Lambda):
    M=cost_matrix(X,Y)
    n,m=M.shape
    dtype=type(Lambda)
    if m==0:
        cost,L,xx,yy=empty_Y_opt(n, Lambda)
        return cost,L
 
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    cost=np.float64(0) # save the optimal cost
    argmin_Y=closest_y_M(M)

    # initial the subproblem
    cost_pre=np.float64(0)
    L_pre=np.empty(0,dtype=np.int64)
    cost_sub_pre=np.float64(0)
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
        i_start=1
        j_start=0
     
    # start loop 
    # 0 denote the case destroy point, 1 denote the case assign to y_jk, 2 denote the case assign to y_jlast+1, 3 denot the case assign to y_jlast 
    cost_book_orig=np.full(4,np.inf,dtype=np.float64)

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
            # cost 2
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost_book[2]=cost+cost_xk_yjlast1
            #cost 3
            cost_xk_yjlast=M[k,j_last]                  
            if cost_xk_yjlast<Lambda and j_start<=m and i_start<k:
                M1=M[i_start:k,j_start:j_last]
                #X1=X[i_start:k]
                #Y1=Y[j_start:j_last]
                L1=L[i_start:k].copy()

        #     # we need the last assign index since we need to retrieve the closest unassigend j                    
                index_adjust(L1,-j_start)
                cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub_a(M1,L1,Lambda)
                cost_book[3]=cost_pre+cost_sub+cost_xk_yjlast                       
                index_adjust(L_sub,j_start)
                index_adjust(L_sub_pre,j_start)

        # find the optimal cost over all 
        cost_book1=cost_book[np.array([3,0,1,2])]
        min_case1=cost_book1.argmin()
        cost=cost_book1.min()
        min_case=np.array([3,0,1,2])[min_case1]
        
        # update problem, if we destroy points, update pre_problem
        if min_case==0:
            L=np.append(L,-1)
            # update previous problem 
            cost_pre=cost
            L_pre=L.copy()
            j_start=j_last+1
            i_start=k+1
           


        elif min_case==1:
            L=np.append(L,jk) #],dtype=np.int64)))
        elif min_case==2:
            L=np.append(L,j_last+1) #],dtype=np.int64)))
        elif min_case==3:
            L=np.concatenate((L_pre,L_sub,np.array([j_last],dtype=np.int64)))
            # update previous problem 
            if L_sub_pre.shape[0]>=1:
                cost_pre=cost_pre+cost_sub_pre
                L_pre=np.concatenate((L_pre,L_sub_pre))
                i_start=i_start+L_sub_pre.shape[0]
                if L_sub_pre.shape[0]>=2:
                    j_start=L_sub_pre[-2]+1

#                i_start,j_start=startindex(L_pre)
            # empty the variable for sub problem
            L_sub=np.empty(0,dtype=np.int64)
            L_sub_pre=np.empty(0,dtype=np.int64)
            cost_sub=np.float64(0)
            cost_sub_pre=np.float64(0)
            
    return cost,L

#@nb.njit()
#@nb.njit(nb.types.Tuple((List,List,nb.int64[:]))(nb.float64[:],nb.float64[:],nb.float64))
def opt_decomposition(X,Y,Lambda):
    M=cost_matrix(X,Y)
    CM=M>=Lambda
    n,m=M.shape
    argmin_Y=closest_y_M(M)
    mass_loss=0
    Occupy=np.full(m,False)
    free_Y=np.empty(0,dtype=np.int64)
    X_list=List()
    Y_list=List()
    k=0
    jk=argmin_Y[k]
    previous_index=0 # this is the index of first subproblem we can merge 
    if CM[k,jk]==1:
        X_list.append(np.array([k]))            
        Y_list.append(np.empty(0,dtype=np.int64))
        free_Y=np.concatenate((free_Y,np.array([-1])))
        previous_index=len(X_list)
        mass_loss+=1
    else:
        X_list.append(np.array([k]))            
        Y_list.append(np.array([jk]))
        free_y=jk-1
        if free_y>=0 and Occupy[free_y]==1:
            free_y=free_Y[-1]
        free_Y=np.concatenate((free_Y,np.array([free_y])))
        Occupy[jk]=True

        
    for k in range(1,n): 
        
        jk=argmin_Y[k]
#        print('previous_index',previous_index)
        if Occupy[jk]==False:
            #
            if CM[k,jk]==True:
                X_list.append(np.array([k]))            
                Y_list.append(np.empty(0,dtype=np.int64))
                free_Y=np.concatenate((free_Y,np.array([-1])))
                previous_index=len(X_list)
            else:
                X_list.append(np.array([k]))            
                Y_list.append(np.array([jk]))
                free_y=jk-1
                if Occupy[free_y]==1:
                    free_y=free_Y[-1]
                free_Y=np.concatenate((free_Y,np.array([free_y])))
                Occupy[jk]=True    
       

        else: # If it is occupied, we should extend the last problem
        
            #check if we need to merge subproblems
            last_free_y=free_Y[-1] # it must be occupied by last problem 
            last_index=free_Y.shape[0]-1
            if last_free_y==-1:
                index_start=previous_index
                
            else:                  
                index_start=np.where(free_Y==last_free_y)[0][0]
            
            # print('free_Y length',len(free_Y))
            # print('X_list length',len(X_list))
            
            # print('last_index',last_index)
            if index_start<last_index:#Then we need to merge all the previous problems 
                merged_X=X_list[last_index]
                merged_Y=Y_list[last_index]
                start=last_index
                for i in range(last_index-1,index_start-1,-1):
                    Y_take=Y_list[i]
                    y_index=Y_take[-1]
                    X_take=X_list[i+1]
                    x_index=X_take[0]
                    if CM[x_index,y_index]:
                        previous_index=i
                        break
                    start=i
                    merged_X=np.concatenate((X_list[i],merged_X))
                    merged_Y=np.concatenate((Y_list[i],merged_Y))
                
                #print('before',free_Y)
                # print('start',start)
                X_list=X_list[0:start]
                X_list.append(merged_X)
                Y_list=Y_list[0:start]
                Y_list.append(merged_Y)
                free_Y=free_Y[0:start+1]
                if start>index_start:
                    free_Y[-1]=-1
                # print(free_Y)

            #extend the last problem      
            # get the last part 
            last_X=X_list[-1] # the last problem must have jk
            last_Y=Y_list[-1]
            last_y=last_Y[-1]
            last_free_y=free_Y[-1]

            
            #extend last_X
            last_X=np.concatenate((last_X,np.array([k])))
            X_list=X_list[0:-1]
            X_list.append(last_X)
        #    print('after',X_list)
            
            # extend last_Y from left 
            x_start=last_X[0]
            if last_free_y>=0 and not CM[x_start,last_free_y]:
                last_Y=np.concatenate((np.array([last_free_y]),last_Y))
                Occupy[last_free_y]=True
               
            #extend last_Y from right
            x_end=last_X[-1]
            if last_y+1<=m-1 and not CM[x_end,last_y+1]:
                last_Y=np.concatenate((last_Y,np.array([last_y+1])))
                Occupy[last_y+1]=True
            elif CM[x_end,last_y+1]:
                last_index=last_y
            Y_list=Y_list[0:-1]
            Y_list.append(last_Y)
            
            # update free spot for last problem
            free_y=last_free_y-1
            if free_y>=0 and Occupy[free_y]:
                free_y=free_Y[-2]
            elif free_y<0:
                free_y=-1
            free_Y[-1]=free_y
    return X_list,Y_list,free_Y

#@nb.njit(nb.types.Tuple((nb.float64,nb.int64[:]))(nb.float64[:],nb.float64[:],nb.float64),parallel=True,fastmath=True)
def opt_1d_v3(X,Y,Lambda):
    X_list,Y_list,free_Y=opt_decomposition(X,Y,Lambda)
    K=len(X_list)
    costs=np.zeros(K,dtype=np.float64)
    plans=List()

    for i in range(K):
        plans.append(np.empty(0,dtype=np.int64))  
    
    for i in nb.prange(K):
        indices_X=X_list[i]
        indices_Y=Y_list[i]
        Xs=X[indices_X]
        Ys=Y[indices_Y]
        cost,L=opt_1d_v2_apro(Xs,Ys,Lambda)
        
        if indices_Y.shape[0]>0:
            index_adjust(L,indices_Y[0])       
        plans[i]=np.concatenate((plans[i],L))
        costs[i]=cost
    L=merge_list(plans)
    cost=np.sum(costs)
    return cost,L



   


@nb.njit(['Tuple((float64,int64[:]))(float64[:],float64[:])'])
def pot(X,Y): 
    n=X.shape[0]
    m=Y.shape[0]
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    dtype=type(X[0])
    cost=0.0 # save the optimal cost
    M=cost_matrix(X,Y)
    
    argmin_Y=closest_y_M(M)
 
    #initial loop:
    k=0
    xk=X[k]
    jk=argmin_Y[k]
    cost_xk_yjk=M[k,jk]

    cost+=cost_xk_yjk
    L=np.append(L,jk)
    for k in range(1,n):
        jk=argmin_Y[k]
        cost_xk_yjk=M[k,jk]
        j_last=L[-1]
    
        #define consistent term     
        if jk>j_last:# No conflict, L[-1] is the j last assig
            cost+=cost_xk_yjk
            L=np.append(L,jk)
        else:
            # this is the case for conflict: 

            # compute the first cost 
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost1=cost+cost_xk_yjlast1
            else:
                cost1=np.inf 
            # compute the second cost 
            i_act,j_act=unassign_y(L)
            if j_act>=0:                        
                L1=np.concatenate((L[0:i_act],np.array([j_act]),L[i_act:]))
                X_indices=arange(0,k+1)
                Y_indices=L1
#                Y_assign=Y[L1]
#                X_assign=X[0:k+1]
                cost2=np.sum(matrix_take(M,X_indices,Y_indices))
#                cost2=np.sum(cost_function(X_assign,Y_assign))
            else:
                cost2=np.inf
            if cost1<cost2:
                cost=cost1
                L=np.append(L,j_last+1)
            elif cost2<=cost1:
                cost=cost2
                L=L1.copy()    
    return cost,L




@nb.njit(['Tuple((float32,int64[:]))(float32[:],float32[:])'])
def pot_32(X,Y): 
    n=X.shape[0]
    m=Y.shape[0]
    L=np.empty(0,dtype=np.int64) # save the optimal plan
    dtype=type(X[0])
    cost=np.float32(0)  # save the optimal cost
    M=cost_matrix(X,Y)    
    argmin_Y=closest_y_M(M)
 
    #initial loop:
    k=0
    xk=X[k]
    jk=argmin_Y[k]
    cost_xk_yjk=M[k,jk]

    cost+=cost_xk_yjk
    L=np.append(L,jk)
    for k in range(1,n):
        jk=argmin_Y[k]
        cost_xk_yjk=M[k,jk]
        j_last=L[-1]
    
        #define consistent term     
        if jk>j_last:# No conflict, L[-1] is the j last assig
            cost+=cost_xk_yjk
            L=np.append(L,jk)
        else:
            # this is the case for conflict: 

            # compute the first cost 
            if j_last+1<=m-1:
                cost_xk_yjlast1=M[k,j_last+1]
                cost1=cost+cost_xk_yjlast1
            else:
                cost1=np.inf 
            # compute the second cost 
            i_act,j_act=unassign_y(L)
            if j_act>=0:                        
                L1=np.concatenate((L[0:i_act],np.array([j_act]),L[i_act:]))
                X_indices=arange(0,k+1)
                Y_indices=L1
#                Y_assign=Y[L1]
#                X_assign=X[0:k+1]
                cost2=np.sum(matrix_take(M,X_indices,Y_indices))
#                cost2=np.sum(cost_function(X_assign,Y_assign))
            else:
                cost2=np.float32(np.inf)
            if cost1<cost2:
                cost=cost1
                L=np.append(L,j_last+1)
            elif cost2<=cost1:
                cost=cost2
                L=L1.copy()    
    return cost,L




# data=torch.load('data.pt')
# X=data['X']
# Y=data['Y']
# X.sort()
# Y.sort()
# Lambda=np.float64(20)
# opt_decomposition(X,Y,Lambda)
# #torch.save(data,'data.pt')

# n=100
# m=n+500
# #Lambda=4
# Lambda=np.float64(2)

# for i in range(1):
#     X=torch.rand(n,dtype=torch.float64)-1.5
#     Y=torch.rand(m,dtype=torch.float64)*2+1.5
#     X=X.sort().values
#     Y=Y.sort().values
#     X=np.float64(np.random.uniform(-5,5,n))
#     Y=np.float64(np.random.uniform(-8,8,m))+5
#     X.sort()
#     Y.sort()

#     cost1,L1=opt_lp(X,Y,Lambda)
#     cost2,L2=opt_1d_v2(X,Y,Lambda)
    
    
#     X_list,Y_list,free_Y=opt_decomposition(X1,Y1,Lambda)
#     cost1,L1=opt_1d_np(X1,Y1,Lambda)
#     cost2,L2=opt_1d_v2(X1,Y1,Lambda)
#     if np.linalg.norm(L1-L2)>=1 and abs(cost1-cost2)>=0.001:
#         print('error')
#         break

# X=np.array([0.47016394, 1.0328217 , 1.3818892 , 1.4757019 , 1.9214936 ,
#        2.1895437 , 2.604464  , 2.7063289 , 2.7503216 , 2.922912  ],
#       dtype=np.float64)

# Y=np.array([-0.93786955, -0.8235371 , -0.658347  , -0.4214544 , -0.3613819 ,
#        -0.11025443, -0.06072737,  0.02043935,  0.11602888,  0.22903188,
#         0.4837798 ,  0.5873061 ,  0.7503131 ], dtype=np.float64)
# X1=torch.from_numpy(X)
# Y1=torch.from_numpy(Y)
# #X1=np.array([0.9788, 2.0892, 2.5528, 2.6119, 3.0876], dtype=np.float64)
# #Y1=np.array([4.1054, 4.5741, 4.6948, 5.3001, 5.4889, 5.7816, 6.1562, 6.3901],dtype=np.float64)
# cost2,L2=opt_1d_v2(X,Y,Lambda)
# cost1,L1=opt_1d_v1(X,Y,Lambda)
# opt_1d_np(X,Y,Lambda)
# X1=np.array([ 6.199362,  10.1927185])


# a,b,c,d=opt_sub_np(X1,Y1,L1,Lambda)

