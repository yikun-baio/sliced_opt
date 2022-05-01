# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:57:34 2022

@author: Yikun Bai
Yikun.bai@Vanderbilt.edu
"""

import numpy as np
import math
from library import *

            
def opt_1d_v1(X,Y,Lambda):
    '''
    Parameters
    ------
    X: array_list shape, shape(n,) or (n,1), X should be shorted
    Y: array_list shape, shape(n,) or (n,1), X should be shorted
    Lambda: scolar
    
    Returns:
    -----
    cost: float, cost of opt problem
    L: list, optimal transportation plan
    
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
        if min_case=='1n':
            L.append(jk)
        elif min_case=='1c':
            L.append(j_last+1)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2c':
            L=L_sub+[j_last]
   
             
    n=len(X)
    m=len(Y)
    L=[]
    cost=0  
    X=list_to_numpy_array(X)
    Y=list_to_numpy_array(Y)
    if m==0:
        cost,L,xx,yy=empty_Y_opt(n,Lambda)
        return cost,L
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


def opt_sub(X1,Y1,L1,Lambda):
    L_sub_pre=[]
    cost_sub_pre=0
    if len(Y1)==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt(X1.shape[0],Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre
    i_act,j_act=unassign_y(L1) # i_act is the index for L_sub, not the original_index
        
    L1=L1[:-1]   
    if len(L1)==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt(X1,Y1,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre
    
    # separate the problem 
    Y1_assign=Y1[L1] # It give the list y(L(i_start)),y(L(i_start+1)),....y(L(k-2)). 
    L1_inact=L1[0:i_act] 
    X1_inact=X1[0:i_act]
    Y1_inact=Y1_assign[0:i_act]
    cost_inact=sum(cost_function(X1_inact,Y1_inact)) # no pair has distance exceeds 2Lambda
    #
    X1_act=X1[i_act:]
    Y1_act=Y1_assign[i_act:]
    L1_act=L1[i_act:]
    

    if len(L1_act)==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt(X1_act,Y1,j_act,Lambda)
        cost_sub=cost_inact+cost_sub      
        L_sub=L1_inact+L_sub
        if -np.inf in L_sub:
            cost_sub_pre=cost_sub 
            L_sub_pre=L_sub.copy()            
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
  
    cost_d1_opt={'cost':np.inf,'i_act_destroy':[],'cost_pre':0}# we will store the best transporation plan which destroies one point 

    
    for l in range(len(X1_act)-1,-1,-1): # this is the index we extract x_jk,y_l-1
        xl1=X1_act[l]
        yl=Y1_act[l-1]
        c_xl1yl=cost_function(xl1, yl)
        X1_act_left=X1_act[0:l]
        X1_act_right=X1_act[l+1:]
        Y1_act_left=Y1_act[0:l]
        Y1_act_right=Y1_act[l:]
        cost_left=sum(cost_function(X1_act_left,Y1_act_left))+Lambda # it used to compute the previous cost and the total cost
        cost_right=sum(cost_function(X1_act_right,Y1_act_right)) 
        cost_d1=cost_left+cost_right        

        if cost_d1<cost_d1_opt['cost']:
            cost_d1_opt['cost']=cost_d1
            cost_d1_opt['cost_pre']=cost_left
            cost_d1_opt['i_destroy']=l # store the index of the destroyed point x_(l+1)

        #print(cost_d1_opt)
        if c_xl1yl>=Lambda:# That is, we will nolonger consider other plan since they will not be optimal. 
            break
    # find the unique plan which preserve all points (if exists)
    cost_d0=None # we will store the unique best transportation cost which does not destroy points

    if l==0 and j_act>=0:# in this case we need to consider the plan preserve all points
        cost_xkyl=cost_function(X1_act[0],Y1[j_act])
        if cost_xkyl<Lambda:
            cost_d0=sum(cost_function(X1_act[1:],Y1_act))+cost_xkyl
    
    # find the optimal cost and plan
    if cost_d0==None or cost_d1_opt['cost']<=cost_d0:
        cost_sub=cost_d1_opt['cost']+cost_inact
        #print(cost_d1_opt)
        i_destroy=cost_d1_opt['i_destroy']        
        L_sub=L1_inact+L1_act[0:i_destroy]+[-np.inf]+L1_act[i_destroy:]
        L_sub_pre=L1_inact+L1_act[0:i_destroy]+[-np.inf]
        cost_sub_pre=cost_inact+cost_d1_opt['cost_pre']
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    else: 
        cost_sub=cost_d0+cost_inact
        L_sub=L1_inact+[j_act]+L1_act
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
        


def opt_1d_v2(X,Y,Lambda):
    def update_problem(min_case):
        nonlocal cost_pre
        nonlocal L_pre
        nonlocal cost_sub_pre
        nonlocal L_sub_pre
        if min_case=='0':
            cost_pre=cost
            L_pre=L.copy()

        elif min_case=='2c' and -np.inf in L_sub_pre:
            cost_pre=cost_pre+cost_sub_pre
            L_pre=L_pre+L_sub_pre
            cost_sub_pre=None
            L_sub_pre=None
           
            
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        nonlocal cost_sub_pre
        nonlocal L_sub_pre
        cost_book={}
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+Lambda
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=cost_function(xk,Y[j_last+1])
                if cost_xk_yjlast1<Lambda:
                    cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2c': # cost for recursive problem                
                cost_xk_yjlast=cost_function(xk,Y[j_last])                
                if cost_xk_yjlast<Lambda and j_start<=m:
                    X1=X[i_start:k]
                    Y1=Y[j_start:j_last]
                    L1=L[i_start:k] # we need the last assign index since we need to retrieve the closest unassigend j                    
                    L1=index_adjust(L1,-j_start)
                    cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub(X1,Y1,L1,Lambda)
                    cost_book[case]=cost_pre+cost_sub+cost_xk_yjlast   
                    L_sub=index_adjust(L_sub,j_start)
                    L_sub_pre=index_adjust(L_sub_pre,j_start)
        
        # update transportation plan L            
     
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            L.append(jk)
        elif min_case=='1c':
            L.append(j_last+1)
        elif min_case=='0':
            L.append(-np.inf)
        elif min_case=='2c':
            L=L_pre+L_sub+[j_last]
        if min_case in ['0','2c']:
            update_problem(min_case)
        
    X=list_to_numpy_array(X)
    Y=list_to_numpy_array(Y)
    n=X.shape[0]
    m=Y.shape[0]
    Lambda=2*Lambda
 
    L=[] # save the optimal plan
    cost=0 # save the optimal cost

    # initial the sub
    cost_pre=0
    L_pre=[]
    cost_sub_pre=None
    L_sub_pre=None

    
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2c'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        i_start,j_start=startindex(L_pre)                
        Yc=Y[j_start:]

        if len(Yc)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2c'}
            cost_plan_select()
            continue

        jk,cost_xk_yjk=closest_y(xk,Yc)
        jk=jk+j_start
        if cost_xk_yjk>=Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2c'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2c'}
            cost_plan_select()
            continue
        
        j_last=L[-1] # index of last y 
        if jk>j_last:# No conflict L[-1]=j_last 
            case_set=case_set-{'1c','2c'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
            
        cost_plan_select()
#        i_start,j_start=update_sub(L,i_start,j_start)
        
      
    return cost,L





    

def opt_sub_np(M1,L1,Lambda):
    n1,m1=M1.shape
    L_sub_pre=np.array([],dtype=int)
    cost_sub_pre=0
    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt_np(n1,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre
    i_act,j_act=unassign_y_np(L1) # i_act is the index for L_sub, not the original_index
    L1=L1[:-1]
    
    if L1.shape[0]==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_np(M1,i_act,j_act,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre


    L1_inact=L1[0:i_act]
    L1x_inact=np.arange(0,i_act)

    cost_inact=sum(M1[L1x_inact,L1_inact])
    if i_act==n1-1:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_np(M1,i_act,j_act,Lambda)
        cost_sub=cost_inact+cost_sub      
        L_sub=np.concatenate([L1_inact,L_sub])
        if -1 in L_sub:
            cost_sub_pre=cost_sub 
            L_sub_pre=L_sub.copy()            
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    # find the optimal d1 plan    
    L1_act=L1[i_act:]
    L1x_act=list(range(i_act,n1))    

    cost_L1=M1[L1x_act[1:],L1_act]
    cost_L2=M1[L1x_act[0:-1],L1_act]
    s=cost_L2.shape[0]
    cost_list=np.concatenate([cost_L1,cost_L2])
    C=np.zeros([2*s,s+1])
    ones=np.ones(s)
    for i in range(s+1):
        C[i:i+s,i]=ones
    cost_d1=np.dot(cost_list,C)
    cost_d1_opt=cost_d1.min()+Lambda
    index_d1_opt=cost_d1.argmin()
    
    # find the optimal d1 plan
    cost_d0=np.inf    
    if j_act>=0:
        cost_d0=cost_d1[0]+M1[i_act,j_act]

    if cost_d1_opt<=cost_d0:
        cost_sub=cost_inact+cost_d1_opt
        L_sub=np.concatenate([L1_inact,np.insert(L1_act,index_d1_opt,-1)])     
        cost_sub_pre=cost_inact+sum(cost_L2[0:index_d1_opt])+Lambda
        L_sub_pre=np.concatenate([L1_inact,np.insert(L1_act[0:index_d1_opt],index_d1_opt,-1)])         
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    else:
        L_sub=np.concatenate([L1_inact,[j_act],L1_act])
        cost_sub=cost_inact+cost_d0
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre



        


def opt_1d_np(X,Y,Lambda):
    '''
    Parameters
    ----------
    X : numpy array n*1 
        DESCRIPTION.
    Y : numpy array m*1
        DESCRIPTION.
    Lambda : flout number
        DESCRIPTION.

    Returns
    -------
    
    cost: optimal value of opt(X,Y;Lambda)
    L: optimal plan of opt(X,Y;Lambda)
    '''
    def update_problem(min_case):
        nonlocal cost_pre
        nonlocal L_pre
        nonlocal cost_sub_pre
        nonlocal L_sub_pre
        if min_case=='0':
            cost_pre=cost
            L_pre=L.copy()

        elif min_case=='2c' and -1 in L_sub_pre:
            cost_pre=cost_pre+cost_sub_pre
            L_pre=np.concatenate([L_pre,L_sub_pre])
            cost_sub_pre=None
            L_sub_pre=None
               
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        nonlocal cost_sub_pre
        nonlocal L_sub_pre

        cost_book={}
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+Lambda
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=M[k,j_last+1]
                #cost_function(xk,Y[j_last+1])
                if cost_xk_yjlast1<Lambda:
                    cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2c': # cost for recursive problem
                cost_xk_yjlast=M[k,j_last]                  
            #    cost_xk_yjlast=cost_function(xk,Y[j_last])
                if cost_xk_yjlast<Lambda and j_start<=m:
                    M1=M[i_start:k,j_start:j_last]
                    L1=L[i_start:k].copy() 
                    # we need the last assign index since we need to retrieve the closest unassigend j                    
                    index_adjust_np(L1,-j_start)
                    cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub_np(M1,L1,Lambda)
                    cost_book[case]=cost_pre+cost_sub+cost_xk_yjlast                       
                    index_adjust_np(L_sub,j_start)
                    index_adjust_np(L_sub_pre,j_start)
                    # print('L_sub is',L_sub)
                    # print('---end---')

        
        # update transportation plan L            
     
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            L=np.concatenate([L,[jk]])
        elif min_case=='1c':
            L=np.concatenate([L,[j_last+1]])
        elif min_case=='0':
            L=np.concatenate([L,[-1]])
        elif min_case=='2c':
            L=np.concatenate([L_pre,L_sub,[j_last]])
        if min_case in ['0','2c']:
            update_problem(min_case)
    
    X=list_to_numpy_array(X)
    Y=list_to_numpy_array(Y)
    M=cost_matrix(X,Y)
    n,m=M.shape
    Lambda=2*Lambda
 
    L=np.empty(0,dtype=int) # save the optimal plan
    cost=0 # save the optimal cost

    # initial the sub
    cost_pre=0
    L_pre=np.empty(0,dtype=int)
    cost_sub_pre=None
    L_sub_pre=None
    
    for k in range (n):
        case_set={'0','1n','1c','2c'} 
        i_start,j_start=startindex(L_pre)
        Yc=Y[j_start:]
        
        if j_start==m: # There is no y, so we destroy point
            cost_end,L_end,xx,yy=empty_Y_opt_np(n-k, Lambda)
            cost=cost+cost_end
            L=np.concatenate([L,L_end])
            return cost,L
        

        jk,cost_xk_yjk=closest_y_np(M,k,j_start)
        jk=jk+j_start
        if cost_xk_yjk>=Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2c'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2c'}
            cost_plan_select()
            continue
        
        j_last=L[-1] # index of last y 
        if jk>j_last:# No conflict L[-1]=j_last 
            case_set=case_set-{'1c','2c'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
    return cost,L

        



def opt_sub_T(M1,X1,Y1,L1,Lambda,device='cuda'):
    n1,m1=M1.shape

    if m1==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=empty_Y_opt_T(n1,Lambda)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre
    i_act,j_act=unassign_y_T(L1) # i_act is the index for L_sub, not the original_index
    L1=L1[:-1]
    if len(L1)==0:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_T(M1,i_act,j_act,Lambda,device)
        return cost_sub,L_sub, cost_sub_pre,L_sub_pre   
    # separate the problem 
    L1_inact=L1[0:i_act] 
    Lx1_inact=torch.arange(0,i_act)
    cost_inact=torch.sum(M1[Lx1_inact,L1_inact])
    L1_act=L1[i_act:]
    if i_act==n1-1:
        cost_sub,L_sub,cost_sub_pre,L_sub_pre=one_x_opt_T(M1,i_act,j_act,Lambda,device)
        cost_sub=cost_inact+cost_sub      
        L_sub=torch.cat([L1_inact,L_sub])
        if -1 in L_sub:
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
    C=torch.zeros([2*s,s+1],device=device)
    ones=torch.ones(s,device=device)
    for i in range(s+1):
        C[i:i+s,i]=ones
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
        L_sub_pre=torch.cat([L1_inact,L1_inact,L1_act[0:index_d1_opt],torch.tensor([-1],device=device)])         
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    else:
        L_sub=torch.cat([L1_inact,j_act.reshape(1),L1_act])
        cost_sub=cost_inact+cost_d0
        cost_sub_pre=torch.tensor(0.0,device=device)
        L_sub_pre=torch.empty(0,device=device,dtype=torch.int)
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre


    
    

def opt_1d_T(X,Y,Lambda,device='cuda'):
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
    def update_problem(min_case):
        nonlocal cost_pre
        nonlocal L_pre
        nonlocal cost_sub_pre
        nonlocal L_sub_pre
        if min_case=='0':
            cost_pre=cost
            L_pre=L.clone()
        elif min_case=='2c' and -1 in L_sub_pre:
            cost_pre=cost_pre+cost_sub_pre
            L_pre=torch.cat([L_pre,L_sub_pre])
            cost_sub_pre=None
            L_sub_pre=None                       
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        nonlocal cost_sub_pre
        nonlocal L_sub_pre
        cost_book={}        
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+Lambda  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=M[k,j_last+1]
                if cost_xk_yjlast1<Lambda:
                    cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2c': # cost for recursive problem              
                cost_xk_yjlast=M[k,j_last]                
                if cost_xk_yjlast<Lambda and j_start<=m:
                    X1=X[i_start:k]
                    Y1=Y[j_start:j_last]
                    L1=L[i_start:k] # we need the last assign index since we need to retrieve the closest unassigend j                    
                    L1=L1-j_start # adjust index
                    M1=M[i_start:k,j_start:j_last]
                    cost_sub,L_sub,cost_sub_pre,L_sub_pre=opt_sub_T(M1,X1,Y1,L1,Lambda)
                    cost_book[case]=cost_pre+cost_sub+cost_xk_yjlast   
                    index_adjust_T(L_sub,j_start)
                    index_adjust_T(L_sub_pre,j_start)
                    #L_sub_pre=torch.where(L_sub_pre<0,-1,L_sub_pre+j_start)
        
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            L=torch.cat([L,jk.reshape(1)])
        elif min_case=='1c':
            L=torch.cat([L,(j_last+1).reshape(1)])
        elif min_case=='0':
            L=torch.cat([L,torch.tensor([-1],device=device)])
        elif min_case=='2c':
            L=torch.cat([L_pre,L_sub,j_last.reshape(1)])
        if min_case in ['0','2c']:
            update_problem(min_case)
            
    M=cost_matrix_T(X,Y,device)
    n,m=M.shape
    device=X.device.type
    Lambda=torch.tensor(Lambda,device=device)*2
    
    L=torch.empty(0,device=device,dtype=torch.int) # save the optimal plan
    cost=torch.tensor(0.0,device=device) # save the optimal cost
    cost_pre=torch.tensor(0.0,device=device)
    L_pre=torch.empty(0,device=device,dtype=torch.int)
    # initial the sub
    cost_sub_pre=None
    L_sub_pre=None
    for k in range (n):
      #  print('L_subpre',L_sub_pre)
        xk=X[k]
        case_set={'0','1n','1c','2c'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        i_start,j_start=startindex_T(L_pre)                
        #Yc=Y[j_start:].clone()
        if j_start==m: # There is no y, so we destroy point
            cost_end,L_end,xx,yy=empty_Y_opt_T(n-k, Lambda)
            cost=cost+cost_end
            L=torch.cat([L,L_end])
            return cost,L
        

        jk,cost_xk_yjk=closest_y_np(M,k,j_start)
        jk=jk+j_start
        if cost_xk_yjk>=Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2c'}
            cost_plan_select()
            continue        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2c'}
            cost_plan_select()
            continue
        j_last=L[-1] # index of last y 
        if jk>j_last:# No conflict L[-1]=j_last 
            case_set=case_set-{'1c','2c'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
    return cost,L




def pot_1d(X,Y): 
    n=len(X)
    m=len(Y)
    L=[] # save the optimal plan
    cost=0 # save the optimal cost
  
    # initial the sub
    X=list_to_numpy_array(X)
    Y=list_to_numpy_array(Y)
    
    for k in range (n):
        xk=X[k]
        jk,cost_xk_yjk=closest_y(xk,Y)
        
        if len(L)==0 or jk>L[-1]:# No conflict, L[-1] is the j last assign
           cost+=cost_xk_yjk
           L.append(jk)
           continue 
        
        # this is the case for conflict: 
        j_last=L[-1]
        # compute the first cost 
        if j_last+1<=m-1:
            ylast1=Y[j_last+1]
            c_xk_yjlast1=cost_function(xk,ylast1)
            cost1=cost+c_xk_yjlast1
        else:
            cost1=numpy.inf 
        
        # compute the second cost 
        i_act,j_act=unassign_y(L)
        if j_act>=0:
            L1=L[0:i_act]+[j_act]+L[i_act:]
            Y_assign=Y[L1]
            X_assign=X[0:k+1]
            cost2=sum(cost_function(X_assign,Y_assign))
        else:
            cost2=np.inf
        if cost1<cost2:
            cost=cost1
            L.append(j_last+1)
        elif cost2<=cost1:
            cost=cost2
            L=L1.copy()    
    return cost,L



Lambda=6
n=15
m=n+3
device='cuda'
for i in range(100):
    X=torch.rand(n,device=device)*(5-0)+0
    Y=torch.rand(m,device=device)*(10-5)+2
    X=X.sort().values
    Y=Y.sort().values
    M=cost_matrix(X,Y)
    cost1,L1=opt_1d_np(X, Y, Lambda)
    cost2,L2=opt_1d_T(X, Y, Lambda)
    if abs(cost1-cost2)>=0.0001:
        print('error!')
        print(L1)
        print(L2)



# X1=np.array([   0, 1, 2])
# Y1=np.array([-1,0, 1]) 
# L1=np.array([1,2,3])

# M1=cost_matrix(X1,Y1)

# n1,m1=M1.shape
# L_sub_pre=np.array([],dtype=int)
# cost_sub_pre=0

# i_act,j_act=unassign_y_np(L1) # i_act is the index for L_sub, not the original_index
# L1=L1[:-1]



# Y1_assign=Y1[L1] # It give the list y(L(i_start)),y(L(i_start+1)),....y(L(k-2)). 
# L1_inact=L1[0:i_act]
# L1x_inact=np.arange(0,i_act)
# X1_inact=X1[0:i_act]
# Y1_inact=Y1_assign[0:i_act]
# cost_inact=sum(M1[L1x_inact,L1_inact])
# X1_act=X1[i_act:]
# Y1_act=Y1_assign[i_act:]
# L1_act=L1[i_act:]
# L1x_act=list(range(i_act,n1))


 
    


# Lambda=5
# X=np.array([ 1.74537179,9.08186784,12.68307632])
# Y=np.array([ 6.65760547, 10.50520207, 24.39995201])
# cost2,L2=opt_1d_np(X,Y,Lambda)

# opt_1d_np(X,Y,Lambda)
# # X1=np.array([ 6.199362,  10.1927185])
# Y1=np.array([5.067703,  6.8166046, 7.4746566])
# L1 =np.array([1,3])

# a,b,c,d=opt_sub_np(X1,Y1,L1,Lambda)

