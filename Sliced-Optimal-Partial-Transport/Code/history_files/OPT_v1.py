# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:12:44 2022

@author: laoba
"""

import numpy
import math
import torch


def OPT_1D_empty_Y(X):
    n=len(X)
    L=[-numpy.inf]*n
    cost=(Lambda1+Lambda2)*n
    return cost,L,cost,L
    

def cost_function(x,y,p=2): 
    V=abs(x-y)**p
    return V


def closest_y(x,Y):
    cost_list=cost_function(x,Y)    
    min_index=cost_list.argmin()
    min_cost=cost_list[min_index]
    return min_index,min_cost


            
def index_adjust(L,j_start=0):
    L=[i+j_start for i in L]
    return L
         

def startindex(L_previous):    
    i_start=len(L_previous)
    j_start=0
    L_assigned=[i for i in L_previous if i>=0]            
    if len(L_assigned)>=1:
        j_lastassinged=max(L_previous)
        j_start=j_lastassinged+1
    return i_start,j_start

def list_to_array(*lst):
    r""" Convert a list if in numpy format """
    if len(lst) > 1:
        return [numpy.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return numpy.array(lst[0]) if isinstance(lst[0], list) else lst[0]
    
        
            
def OPT_1D_v1(X,Y):    
    def cost_plan_select():
        nonlocal L
        nonlocal cost
        cost_book={}    
        #select the best cost
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+(Lambda1+Lambda2)  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=cost_function(xk,Y[j_last+1])
                cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2': # cost for recursive problem
            #print('conflict')
                cost_xk_yjlast=cost_function(xk,Y[j_last])
                if cost_xk_yjlast<(Lambda1+Lambda2):            
                    X1=X[0:k]
                    Y1=Y[0:j_last]
                    cost_previous,L1=OPT_1D_v1(X1,Y1)
                    cost_book[case]=cost_previous+cost_xk_yjlast

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
            L=L1+[j_last]


             
    n=len(X)
    m=len(Y)
    L=[]
    cost=0  
    if m==0:
        cost,L,xx,yy=OPT_1D_empty_Y(X)
        return cost,L
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} 
        # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively
        
        

        jk,cost_xk_yjk=closest_y(xk,Y)


        if cost_xk_yjk>=(Lambda1+Lambda2): # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue

        j_last=max(L) # index of last aligned y
        print(L)
        if jk>j_last:# No conflict
            case_set=case_set-{'1c','2'}
        elif jk<=j_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
    return cost,L






def one_point_problem(X_sub,Y_sub,j_active):            
    if j_active<0:
        return Lambda1+Lambda2,[-numpy.inf],Lambda1+Lambda2,[-numpy.inf]
    elif j_active>=0:
        yl=Y_sub[j_active]
        xk=X_sub[0]
        c_xkyjk=cost_function(xk,yl)
        if c_xkyjk>=Lambda1+Lambda2:
            return Lambda1+Lambda2,[-numpy.inf],Lambda1+Lambda2,[-numpy.inf]
        elif c_xkyjk<Lambda1+Lambda2:
            return c_xkyjk,[j_active],0,[]



def retrieve_unassigned_y(L_sub):        
    j_last_assigned=L_sub[-1]
    i_last_assigned=len(L_sub)-1 # this is the value of k-i_start
    
    if j_last_assigned<0:
        print('a bug')
        return None

    for i in range(0,j_last_assigned+1):
        j=j_last_assigned-i
        i=i_last_assigned-i+1
        if j not in L_sub:
            return i,j

    return 0,-numpy.inf



                    
def OPT_1D_v2_1(X,Y): 
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
        cost_sub_previous=None
        L_sub_previous=None
        cost_book={}
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+(Lambda1+Lambda2)  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=cost_function(xk,Y[j_last+1])
                cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2': # cost for recursive problem                
                cost_xk_yjlast=cost_function(xk,Y[j_last])
                if cost_xk_yjlast<(Lambda1+Lambda2) and j_start<=m:
                    X_sub=X[i_start:k]
                    Y_sub=Y[j_start:j_last]
                    cost_sub,L_sub,cost_sub_previous,L_sub_previous=OPT_1D_v2_1(X_sub,Y_sub)
                    cost_book[case]=cost_previous+cost_sub+cost_xk_yjlast   
                    L_sub=index_adjust(L_sub,j_start)
                    L_sub_previous=index_adjust(L_sub_previous,j_start)  
                    
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
            L=L_previous+L_sub+[j_last]
        if min_case in {'0','2'}:
            update_problem(min_case)

        


    n=len(X)
    m=len(Y)
    
    L=[] # save the optimal plan
    cost=0 # save the optimal cost    
    
    #For sub
    cost_previous=0
    L_previous=[]
    cost_sub_previous=0
    L_sub_previous=[]
    
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        i_start,j_start=startindex(L_previous)         
        Y1=Y[j_start:]

        if len(Y1)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue

        jk,cost_xk_yjk=closest_y(xk,Y1)
        jk=jk+j_start
        if cost_xk_yjk>=Lambda1+Lambda2: # closest distance is 2Lambda, then we destroy the point
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

    
    return cost,L,cost_previous,L_previous






    


    


def OPT_sub(X_sub,Y_sub,L_sub):
    L_sub_previous=[]
    cost_sub_previous=0
    if len(Y_sub)==0: # initial case, empty Y
        cost_sub,L_sub,cost_sub_previous,L_sub_previous=OPT_1D_empty_Y(X_sub)
        return cost_sub,L_sub, cost_sub_previous,L_sub_previous
    i_active,j_active=retrieve_unassigned_y(L_sub) # i_active is the index for L_sub, not the original_index
   

        
    L_sub=L_sub[:-1]
    
    if len(L_sub)==0:

        cost_sub,L_sub,cost_sub_previous,L_sub_previous=one_point_problem(X_sub,Y_sub,j_active)
        return cost_sub,L_sub, cost_sub_previous,L_sub_previous
   
    
    
    # separate the problem 
    Y_sub_assigned=Y_sub.take(L_sub) # It give the list y(L(i_start)),y(L(i_start+1)),....y(L(k-2)). 
    L_sub_inactive=L_sub[0:i_active] 
    X_sub_inactive=X_sub[0:i_active]
    Y_sub_inactive=Y_sub_assigned[0:i_active]
    
    cost_inactive=sum(cost_function(X_sub_inactive,Y_sub_inactive)) # no pair has distance exceeds 2Lambda
    
    #
    X_sub_active=X_sub[i_active:]
    Y_sub_active=Y_sub_assigned[i_active:]
    L_sub_active=L_sub[i_active:]
    


    if len(L_sub_active)==0:
          
        cost_sub,L_sub,cost_sub_previous,L_sub_previous=one_point_problem(X_sub_active,Y_sub,j_active)
        cost_sub=cost_inactive+cost_sub      
        L_sub=L_sub_inactive+L_sub
        if -numpy.inf in L_sub:
            cost_sub_previous=cost_sub 
            L_sub_previous=L_sub.copy()            
        return cost_sub,L_sub,cost_sub_previous,L_sub_previous
  
    cost_destroy_onepoint={'cost':numpy.inf,'i_active_destroy':[],'cost_previous':0}# we will store the best transporation plan which destroies one point 

    
    for l in range(len(X_sub_active)-1,-1,-1): # this is the index we extract x_jk,y_l-1
 
        xl1=X_sub_active[l]
        yl=Y_sub_active[l-1]
        #print('x select is'+str(xl1))
        #print('y select is'+str(yl))
        c_xl1yl=cost_function(xl1, yl)
        X_sub_active_left=X_sub_active[0:l]
        X_sub_active_right=X_sub_active[l+1:]
        Y_sub_active_left=Y_sub_active[0:l]
        Y_sub_active_right=Y_sub_active[l:]
        cost_left=sum(cost_function(X_sub_active_left,Y_sub_active_left))+Lambda1+Lambda2 # it used to compute the previous cost and the total cost
        cost_right=sum(cost_function(X_sub_active_right,Y_sub_active_right)) 
        cost_destroy_xl1=cost_left+cost_right        

        if cost_destroy_xl1<cost_destroy_onepoint['cost']:
            cost_destroy_onepoint['cost']=cost_destroy_xl1
            cost_destroy_onepoint['cost_previous']=cost_left
            cost_destroy_onepoint['i_destroy']=l # store the index of the destroyed point x_(l+1)

        #print(cost_destroy_onepoint)
        if c_xl1yl>=Lambda1+Lambda2:# That is, we will nolonger consider other plan since they will not be optimal. 
            break
    # find the unique plan which preserve all points (if exists)
    cost_destroy_zeropoint=None # we will store the unique best transportation cost which does not destroy points

    if l==0 and j_active>=0:# in this case we need to consider the plan preserve all points
        cost_xkyl=cost_function(X_sub_active[0],Y_sub[j_active])
        if cost_xkyl<Lambda1+Lambda2:
            cost_destroy_zeropoint=sum(cost_function(X_sub_active[1:],Y_sub_active))+cost_xkyl
    
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
        
    


def OPT_1D_v3(X,Y):
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
                cost_book[case]=cost+Lambda1+Lambda2  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yjk
            elif case=='1c' and j_last+1<=m-1: # there exists right point
                cost_xk_yjlast1=cost_function(xk,Y[j_last+1])
                if cost_xk_yjlast1<Lambda1+Lambda2:
                    cost_book[case]=cost+cost_xk_yjlast1
            elif case=='2': # cost for recursive problem                
                cost_xk_yjlast=cost_function(xk,Y[j_last])                
                if cost_xk_yjlast<Lambda1+Lambda2 and j_start<=m:
                    X_sub=X[i_start:k]
                    Y_sub=Y[j_start:j_last]
                    L_sub_0=L[i_start:k] # we need the last assigned index since we need to retrieve the closest unassigend j                    
                    L_sub_0=index_adjust(L_sub_0,-j_start)
                    cost_sub,L_sub,cost_sub_previous,L_sub_previous=OPT_sub(X_sub,Y_sub,L_sub_0)
                    cost_book[case]=cost_previous+cost_sub+cost_xk_yjlast   
                    L_sub=index_adjust(L_sub,j_start)
                    L_sub_previous=index_adjust(L_sub_previous,j_start)
                



            
            

                    
        

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
            L=L_previous+L_sub+[j_last]
        if min_case in ['0','2']:

            # if min_case=='2':
            #     print('L_previous is',L_previous)
            #     print('L_sub_0 is',L_sub_0)
            #     print('L_sub is',L_sub)
            #     print('L_sub_previous is',L_sub_previous)
            
            update_problem(min_case)
            # if min_case=='2':
            #     print('after L_previous is',L_previous)

        
    


    n=len(X)
    m=len(Y)
 
    L=[] # save the optimal plan
    cost=0 # save the optimal cost

    # initial the sub
    cost_previous=0
    L_previous=[]
    cost_sub_previous=0
    L_sub_previous=[]

    
    for k in range (n):

      
        xk=X[k]
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        i_start,j_start=startindex(L_previous)                
        Y1=Y[j_start:]

        if len(Y1)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue

        jk,cost_xk_yjk=closest_y(xk,Y1)
        jk=jk+j_start
        if cost_xk_yjk>=Lambda1+Lambda2: # closest distance is 2Lambda, then we destroy the point
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


