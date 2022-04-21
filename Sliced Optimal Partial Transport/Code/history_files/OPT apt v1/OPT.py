# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:57:34 2022

@author: Yikun Bai
Yikun.bai@Vanderbilt.edu
"""

import numpy
import math
from library import *

def empty_Y_OPT(n,Lambda):
    L=[-numpy.inf]*n
    cost=(2*Lambda)*n
    return cost,L,cost,L


def one_point_OPT(M_sub,i_active,j_active,Lambda):            
    if j_active<0:
        return 2*Lambda,[-numpy.inf],2*Lambda,[-numpy.inf]
    elif j_active>=0:
        c_xkyjk=M_sub[i_active,j_active] 
        if c_xkyjk>=2*Lambda:
            return 2*Lambda,[-numpy.inf],2*Lambda,[-numpy.inf]
        elif c_xkyjk<2*Lambda:
            return c_xkyjk,[j_active],0,[]

 
def OPT_1D_v1(X,Y,Lambda):  
    def cost_plan_select():
        nonlocal L
        nonlocal cost
        cost_book={}    
        #select the best cost
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+(2*Lambda)  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=cost_function(xk,Y[j_last+1])
                cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2': # cost for recursive problem
            #print('conflict')
                cost_xk_yjlast=cost_function(xk,Y[j_last])
                if cost_xk_yjlast<(2*Lambda):            
                    X1=X[0:k]
                    Y1=Y[0:j_last]
                    cost_sub,L_sub=OPT_1D_v1(X1,Y1,Lambda)
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
        elif min_case=='2':
            L=L_sub+[j_last]
   
             
    n=len(X)
    m=len(Y)
    L=[]
    cost=0  
    if m==0:
        cost,L,xx,yy=empty_Y_OPT(n,Lambda)
        return cost,L
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} 
        # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively
        
        jk,cost_xk_yjk=closest_y(xk,Y)

        if cost_xk_yjk>=(2*Lambda): # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue

        j_last=max(L) # index of last aligned y
        if jk>j_last:# No conflict
            case_set=case_set-{'1c','2'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
    return cost,L
        
            
def OPT_1D_v1_1(M,Lambda):  
    def cost_plan_select():
        nonlocal L
        nonlocal cost
        cost_book={}    
        #select the best cost
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+(2*Lambda)  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=M[k,j_last+1]
                cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2': # cost for recursive problem
            #print('conflict')
                cost_xk_yjlast=M[k,j_last]
                if cost_xk_yjlast<(2*Lambda):            
                    M_sub=M[0:k,0:j_last]
                    cost_sub,L_sub=OPT_1D_v1_1(M_sub,Lambda)
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
        elif min_case=='2':
            L=L_sub+[j_last]
   
             
    n,m=M.shape
    L=[]
    cost=0  
    if m==0:
        cost,L,xx,yy=empty_Y_OPT(n,Lambda)
        return cost,L
    for k in range (n):
        case_set={'0','1n','1c','2'} 
        # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively
        
        jk,cost_xk_yjk=closest_y_C(k,M)

        if cost_xk_yjk>=(2*Lambda): # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue

        j_last=max(L) # index of last aligned y
        if jk>j_last:# No conflict
            case_set=case_set-{'1c','2'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
    return cost,L





def OPT_sub(M_sub,L_sub_0,Lambda):
    print('L_sub_0 is', L_sub_0)
    L_sub_previous=[]
    cost_sub_previous=0
    n_sub,m_sub=M_sub.shape
    if m_sub==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_previous,L_sub_previous=empty_Y_OPT(n_sub,Lambda)
        return cost_sub,L_sub, cost_sub_previous,L_sub_previous
    i_active,j_active=retrieve_unassigned_y(L_sub_0) # i_active is the index for L_sub, not the original_index

        

    L_sub_0=L_sub_0[:-1]
    if len(L_sub_0)==0:
        cost_sub,L_sub,cost_sub_previous,L_sub_previous=one_point_OPT(M_sub,i_active,j_active,Lambda)
        return cost_sub,L_sub, cost_sub_previous,L_sub_previous
   
    Lx_sub=list(range(0,n_sub))
    # seperate the problem  
    #Y_sub_assigned=Y_sub[L_sub] # It give the list y(L(i_start)),y(L(i_start+1)),....y(L(k-2)).
    Lx_sub_inactive=Lx_sub[0:i_active]
    L_sub_inactive=L_sub_0[0:i_active]
    cost_inactive=sum(M_sub[Lx_sub_inactive,L_sub_inactive]) # no pair has distance exceeds 2Lambda

    
    Lx_sub_active=Lx_sub[i_active:]
    L_sub_active=L_sub_0[i_active:]
    #M_sub_active=M_sub[i_active:,j_active:]

    if len(L_sub_active)==0:
        cost_sub,L_sub,cost_sub_previous,L_sub_previous=one_point_OPT(M_sub,i_active,j_active,Lambda)
        cost_sub=cost_inactive+cost_sub      
        L_sub=L_sub_inactive+L_sub
        if -numpy.inf in L_sub:
            cost_sub_previous=cost_sub 
            L_sub_previous=L_sub.copy()            
        return cost_sub,L_sub,cost_sub_previous,L_sub_previous
  
    cost_destroy_onepoint={'cost':numpy.inf,'i_active_destroy':[],'cost_previous':0}# we will store the best transporation plan which destroies one point 
    
    
    for l in range(len(Lx_sub_active)-1,-1,-1): # this is the index we extract x_jk,y_l-1 
        #xl=X_sub_active[l]
        #yl1=Y_sub_active[l-1]
        Lx_sub_active_left=Lx_sub_active[0:l]
        Lx_sub_active_right=Lx_sub_active[l+1:]
        L_sub_active_left=L_sub_active[0:l]
        L_sub_active_right=L_sub_active[l:]
        
        cost_left=sum(M_sub[Lx_sub_active_left,L_sub_active_left])+2*Lambda # it used to compute the previous cost and the total cost
        cost_right=sum(M_sub[Lx_sub_active_right,L_sub_active_right]) 
        cost_destroy_xl1=cost_left+cost_right        

        if cost_destroy_xl1<cost_destroy_onepoint['cost']:
            cost_destroy_onepoint['cost']=cost_destroy_xl1
            cost_destroy_onepoint['cost_previous']=cost_left
            cost_destroy_onepoint['i_destroy']=l # store the index of the destroyed point x_(l+1)

        #print(cost_destroy_onepoint)
        c_xl1yl=M_sub[Lx_sub_active[l],L_sub_active[l-1]]

        if c_xl1yl>=2*Lambda:# That is, we will nolonger consider other plan since they will not be optimal. 
            break
    # find the unique plan which preserve all points (if exists)
    cost_destroy_zeropoint=None # we will store the unique best transportation cost which does not destroy points

    if l==0 and j_active>=0:# in this case we need to consider the plan preserve all points
        cost_xkyl=M_sub[i_active,j_active]
        if cost_xkyl<2*Lambda:
            cost_destroy_zeropoint=sum(M_sub[Lx_sub_active[1:],L_sub_active])+cost_xkyl
    
    # find the optimal cost and plan
    
    if cost_destroy_zeropoint==None or cost_destroy_onepoint['cost']<=cost_destroy_zeropoint:        
        cost_sub=cost_destroy_onepoint['cost']+cost_inactive
        #print(cost_destroy_onepoint)
        i_destroy=cost_destroy_onepoint['i_destroy']        
        L_sub=L_sub_inactive+L_sub_active[0:i_destroy]+[-numpy.inf]+L_sub_active[i_destroy:]
        L_sub_previous=L_sub_inactive+L_sub_active[0:i_destroy]+[-numpy.inf]
        cost_sub_previous=cost_inactive+cost_destroy_onepoint['cost_previous']
        return cost_sub,L_sub,cost_sub_previous,L_sub_previous


    else: 
        cost_sub=cost_destroy_zeropoint+cost_inactive
        L_sub=L_sub_inactive+[j_active]+L_sub_active
        return cost_sub,L_sub,cost_sub_previous,L_sub_previous
        
    


def OPT_1D_v3(M,Lambda):
    def update_problem(min_case):
        nonlocal cost_previous
        nonlocal L_previous
        if min_case=='0':
            cost_previous=cost
            L_previous=L.copy()
        elif min_case=='2' and -numpy.inf in L_sub_previous:
            cost_previous=cost_previous+cost_sub_previous
            L_previous=L_previous+L_sub_previous
           
            
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        nonlocal cost_sub_previous
        nonlocal L_sub_previous
        cost_sub_previous=0
        L_sub_previous=[]
        cost_book={}

        
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+2*Lambda  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=M[k,j_last+1]
                if cost_xk_yjlast1<2*Lambda:
                    cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2': # cost for recursive problem                
                cost_xk_yjlast=M[k,j_last]                
                if cost_xk_yjlast<2*Lambda and j_start<=m:
                   

                    L_sub_0=L[i_start:k] # we need the last assigned index since we need to retrieve the closest unassigend j
                    M_sub=M[i_start:k,j_start:j_last] 
                   
                    L_sub_0=index_adjust(L_sub_0,-j_start)
                    cost_sub,L_sub,cost_sub_previous,L_sub_previous=OPT_sub(M_sub,L_sub_0,Lambda)
                    cost_book[case]=cost_previous+cost_sub+cost_xk_yjlast   
                    L_sub=index_adjust(L_sub,j_start)
                    L_sub_previous=index_adjust(L_sub_previous,j_start)
                    print('L_sub_0 is',L_sub_0)
                    print('M_sub is',M_sub)
                    print('cost_sub is',cost_sub)
                    print('cost_sub_previous is',cost_sub_previous)
                    print('L_sub is',L_sub)
                    print('L_sub_previous is',L_sub_previous)
        
                


        # update transportation plan L
         
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            L.append(jk)
        elif min_case=='1c':
            L.append(j_last+1)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2':
            print('k is',k)
            print('L is',L)
            print('jlast is',j_last)
    
    
            print('min case is',min_case)
            print('L is', L)
        

 

        
    

    n,m=M.shape

 
    L=[] # save the optimal plan
    cost=0 # save the optimal cost

    # initial the sub
    cost_previous=0
    L_previous=[]
    cost_sub_previous=0
    L_sub_previous=[]

    
    for k in range (n):
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        i_start,j_start=startindex(L_previous)
        
        M2=M[:,j_start:]
        if M2.shape[1]==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        jk,cost_xk_yjk=closest_y_C(k,M2)
        jk=jk+j_start
        if cost_xk_yjk>=2*Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue
        
        j_last=L[-1] # index of last y 
        

        if jk>j_last:# No conflict L[-1]=j_last 
            case_set=case_set-{'1c','2'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
            
        cost_plan_select()
#        i_start,j_start=update_sub(L,i_start,j_start)
        
      
    return cost,L



def POT_1D(X,Y): 
    n=len(X)
    m=len(Y)
 
    L=[] # save the optimal plan
    cost=0 # save the optimal cost
  
    # initial the sub
    problem_previous={
        'L_previous':[],
        'cost_previous':0,
        }
    
    for k in range (n):
        xk=X[k]
        jk,cost_xk_yjk=closest_y(xk,Y)
        
        if len(L)==0 or jk>L[-1]:# No conflict, L[-1] is the j last assigned
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
        i_active,j_active=retrieve_unassigned_y(L)
        if j_active>=0:
            L1=L[0:i_active]+[j_active]+L[i_active:]
            Y_assigned=Y[L1]
            X_assigned=X[0:k+1]
            cost2=sum(cost_function(X_assigned,Y_assigned))
        else:
            cost2=numpy.inf
        if cost1<cost2:
            cost=cost1
            L.append(j_last+1)
        elif cost2<=cost1:
            cost=cost2
            L=L1.copy()    
    return cost,L

Lambda=3
for i in range(100):
    X=numpy.random.uniform(0,5,3)
    Y=numpy.random.uniform(2,10,5)+1
    X.sort()
    Y.sort()
    M=cost_matrix(X,Y)

    cost1,L1=OPT_1D_v1_1(M,Lambda)
    cost2,L2=OPT_1D_v3(M,Lambda)
    if abs(cost1-cost2)>0.1:
        print(X)
        print(Y)
        print(abs(cost1-cost2))
X=numpy.array([1, 4.5,  5])
Y=numpy.array([1, 3.5, 5, 6])
M=cost_matrix(X,Y)
cost1,L1=OPT_1D_v1_1(M,Lambda)
cost2,L2=OPT_1D_v3(M,Lambda)
print(L1)
print(L2)




M_sub=numpy.array( [[ 0. ,6.25],[12.25 , 1.  ]])
L_sub_0=[0,2]
cost,L,cost_p,L_p=OPT_sub(M_sub,L_sub_0,Lambda)

        

