# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:47:54 2022

@author: laoba
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:57:34 2022

@author: Yikun Bai
Yikun.bai@Vanderbilt.edu
"""

import numpy
import math
import torch


def OPT_1D_empty_Y(X):
    n=len(X)
    L=[-numpy.inf]*n
    cost=2*Lambda*n
    return cost,L
    

def cost_function(x,y,p=2): 
    V=abs(x-y)**p
    return V



def closest_y(x,Y):
    cost_list=cost_function(x,Y)    
    min_index=cost_list.argmin()
    min_cost=cost_list[min_index]
    return min_index,min_cost



            
def index_adjust(L,y_startindex=0):
    for i in range(len(L)):
        if L[i]>=0:
            L[i]=L[i]+y_startindex
        

def startindex(L_previous):
    if len(L_previous)>=1 and L_previous[-1]>=0:
        print('a bug in L_previous')
    
    x_startindex=len(L_previous)
    y_startindex=0
    L_assigned=[i for i in L_previous if i>=0]            
    if len(L_assigned)>=1:
        y_index_lastassinged=max(L_previous)
        y_startindex=y_index_lastassinged+1
    return x_startindex,y_startindex
    
        
            
def OPT_1D_v1(X,Y):    
    def cost_plan_select():
        nonlocal L
        nonlocal cost
        cost_book={}    
        #select the best cost
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+2*Lambda  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yl
            elif case=='1c' and y_index_last+1<=m-1: # there exists right point
                cost_xk_ylast1=cost_function(xk,Y[y_index_last+1])
                if cost_xk_ylast1<2*Lambda:
                    cost_book[case]=cost+cost_xk_ylast1
            elif case=='2': # cost for recursive problem
            #print('conflict')
                cost_xk_ylast=cost_function(xk,Y[y_index_last])
                if cost_xk_ylast<2*Lambda:            
                    X1=X[0:k]
                    Y1=Y[0:y_index_last]
                    cost_previous,L1=OPT_1D_v1(X1,Y1)

                    cost_book[case]=cost_previous+cost_xk_ylast
        # Update L 
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            L.append(l)
        elif min_case=='1c':
            L.append(y_index_last+1)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2':
            L=L1+[y_index_last]
            
    n=len(X)
    m=len(Y)
    L=[]
    cost=0     
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively
        
        if m==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue

        l,cost_xk_yl=closest_y(xk,Y)
        if cost_xk_yl>=2*Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue

        y_index_last=max(L) # index of last aligned y 
        x_index_last=L.index(y_index_last) # index of Last allocated x
        if l>y_index_last:# No conflict
            case_set=case_set-{'1c','2'}
        elif l<=y_index_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()

    return cost,L

def OPT_1D_v2(X,Y):
    def update_subproblem():
        nonlocal problem_previous   
        # update the subproblem
        y_index_current=L[-1]
        if y_index_current<0:
            problem_previous['cost_previous']=cost
            problem_previous['L_previous']=L.copy()
            
            
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        # select the best cost
        cost_book={}
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+2*Lambda  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yl
            elif case=='1c' and y_index_last+1<=m-1: # there exists right point
                cost_xk_ylast1=cost_function(xk,Y[y_index_last+1])
                if cost_xk_ylast1<2*Lambda:
                    cost_book[case]=cost+cost_xk_ylast1
            elif case=='2': # cost for recursive problem
            #print('conflict')

                cost_xk_ylast=cost_function(xk,Y[y_index_last])
                x_startindex,y_startindex=startindex(problem_previous['L_previous'])
                if cost_xk_ylast<2*Lambda and y_startindex<=m:

                    
                    X_sub=X[x_startindex:k]
                    Y_sub=Y[y_startindex:y_index_last]
                    cost_subproblem,L_subproblem=OPT_1D_v2(X_sub,Y_sub)   
                    cost_book[case]=problem_previous['cost_previous']+cost_subproblem+cost_xk_ylast
                    index_adjust(L_subproblem,y_startindex)
            
                    

        
        # Update optimal plan L 
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]

        if min_case=='1n':
            y_index=l # we make the problem smaller and need to modify th index
            L.append(y_index)
        elif min_case=='1c':
            y_index=y_index_last+1
            L.append(y_index)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2':
            L=problem_previous['L_previous']+L_subproblem+[y_index_last]
        update_subproblem()

    


    n=len(X)
    m=len(Y)
    
    L=[] # save the optimal plan
    cost=0 # save the optimal cost

    
    #for problem dividing 
    problem_previous={'cost_previous':0,
                      'L_previous':[],
                      }
    



    
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        x_startindex,y_startindex=startindex(problem_previous['L_previous'])
        Y1=Y[y_startindex:]
        
        if len(Y1)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
           


        l,cost_xk_yl=closest_y(xk,Y1)
        l=l+y_startindex
        if cost_xk_yl>=2*Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue


        
        y_index_last=L[-1] # index of last y 

  
       # here y_index_last_aligned=y_index_last
        if l>y_index_last or y_index_last<0:# No conflict L[-1]=y_index_last 
            case_set=case_set-{'1c','2'}
        elif l<=y_index_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()

    
    return cost,L 




def update_problem(cost,L,problem_previous,min_case=None,subproblem_previous=None):
    if problem_previous==None and L[-1]<0:
        problem_previous={}
        min_case='0'

    if min_case=='0':
        problem_previous['L_previous']=L.copy()
        problem_previous['cost_previous']=cost
        return problem_previous
     
    elif min_case=='2' and subproblem_previous!=None and -numpy.inf in subproblem_previous['L_previous']:
        problem_previous['L_previous']=problem_previous['L_previous']+subproblem_previous['L_previous']
        problem_previous['cost_previous']=problem_previous['cost_previous']+subproblem_previous['cost_previous']
        return problem_previous
             
    return None


                    
def OPT_1D_v2_1(X,Y): 
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        nonlocal problem_previous
        #nonlocal subproblem_previous
        # select the best cost
        subproblem_previous=None
        cost_book={}
        
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+2*Lambda  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yl
            elif case=='1c' and y_index_last+1<=m-1: # there exists right point
                cost_xk_ylast1=cost_function(xk,Y[y_index_last+1])
                if cost_xk_ylast1<2*Lambda:
                    cost_book[case]=cost+cost_xk_ylast1
            elif case=='2': # cost for recursive problem                
                cost_xk_ylast=cost_function(xk,Y[y_index_last])
                x_startindex,y_startindex=startindex(problem_previous['L_previous'])
                if cost_xk_ylast<2*Lambda and y_startindex<=m:
                    X_sub=X[x_startindex:k]
                    Y_sub=Y[y_startindex:y_index_last]
                    cost_subproblem,L_subproblem,subproblem_previous=OPT_1D_v2_1(X_sub,Y_sub)
                    cost_book[case]=problem_previous['cost_previous']+cost_subproblem+cost_xk_ylast   
                    # update index
                    index_adjust(L_subproblem,y_startindex)
                    if subproblem_previous!=None:
                        index_adjust(subproblem_previous['L_previous'],y_startindex)
    
                    
       # update transportation plan L            
    
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            y_index=l # we make the problem smaller and need to modify th index
            L.append(y_index)
        elif min_case=='1c':
            y_index=y_index_last+1
            L.append(y_index)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2':
            L=problem_previous['L_previous']+L_subproblem+[y_index_last]
            
        # update the subproblem
        if min_case in {'0','2'}:
            if L[-1]>=0 and min_case!='2':
                print('min case is '+str(min_case))
                print ('there is a bug in update problem')
                print('L is '+str(L))
                print('X is '+str(X))
                print('Y is'+str(Y))
                print('k is'+str(k))

                
            update_problem(cost,L,problem_previous,min_case,subproblem_previous)
        


    n=len(X)
    m=len(Y)
    
    L=[] # save the optimal plan
    cost=0 # save the optimal cost

    
    
    #For subproblem
    problem_previous={
        'cost_previous':0,
        'L_previous':[],
        }
    #subproblem_previous={
    #    'cost_previous':0,
    #    'L_previous':[],
    #    }


    
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        x_startindex,y_startindex=startindex(problem_previous['L_previous'])                
        Y1=Y[y_startindex:]

        if len(Y1)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue

        l,cost_xk_yl=closest_y(xk,Y1)
        l=l+y_startindex
        if cost_xk_yl>=2*Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue
        
        y_index_last=L[-1] # index of last y 
        

        if l>y_index_last:# No conflict L[-1]=y_index_last 
            case_set=case_set-{'1c','2'}
        elif l<=y_index_last:# conflict
            case_set=case_set-{'1n'}
            
        cost_plan_select()
#        x_startindex,y_startindex=update_subproblem(L,x_startindex,y_startindex)

    
    return cost,L,problem_previous



def retrieve_unassigned_y(L_sub):        
    y_index_last_assigned=L_sub[-1]
    x_index_last_assigned=len(L_sub)-1 # this is the value of k-x_startindex
    
    if y_index_last_assigned<0:
        print('a bug')
        return None

    for i in range(0,y_index_last_assigned+1):
        y_index=y_index_last_assigned-i
        x_index=x_index_last_assigned-i+1
        if y_index not in L_sub:
            return x_index,y_index

    return 0,-numpy.inf


    


def one_point_problem(X_sub,Y_sub,y_index):

            
    if y_index<0:
        return 2*Lambda,[-numpy.inf]
    elif y_index>=0:
        yl=Y_sub[y_index]
        xk=X_sub[0]
        c_xkyl=cost_function(xk,yl)
        if c_xkyl>=2*Lambda:
            return 2*Lambda,[-numpy.inf]
        elif c_xkyl<2*Lambda:
            return c_xkyl,[y_index]



    


def OPT_subproblem(X_sub,Y_sub,L_sub):
    subproblem_previous=None
    if len(Y_sub)==0: # initial case, empty Y
        cost_subproblem,L_subproblem=OPT_1D_empty_Y(X_sub)
        subproblem_previous=update_problem(cost_subproblem,L_subproblem)
        return cost_subproblem,L_subproblem,subproblem_previous


    x_index,y_index=retrieve_unassigned_y(L_sub) # x_index is the index for L_sub, not the original_index
   
    
    if x_index<0:
        print('in OPT subproblem x_index is error'+str(x_index))
        print('L_sub is'+str(L_sub))
        
    L_sub=L_sub[:-1]
    
    if len(L_sub)==0:
        if len(X_sub)>=2:
            print(' a bug in one_point_problem')
            print(X_sub)

        cost_subproblem,L_subproblem=one_point_problem(X_sub,Y_sub,y_index)
        subproblem_previous=update_problem(cost_subproblem,L_subproblem)
        return cost_subproblem,L_subproblem,subproblem_previous
   
    
    
    # separate the problem 
    Y_sub_assigned=Y_sub.take(L_sub) # It give the list y(L(x_startindex)),y(L(x_startindex+1)),....y(L(k-2)). Recall that we assign L(k-1) to x(k) in this case.
    L_sub_inactive=L_sub[0:x_index] 
    X_sub_inactive=X_sub[0:x_index]
    Y_sub_inactive=Y_sub_assigned[0:x_index]
    
    cost_inactive=sum(cost_function(X_sub_inactive,Y_sub_inactive)) # no pair has distance exceeds 2Lambda
    
    
    #
    X_sub_active=X_sub[x_index:]
    Y_sub_active=Y_sub_assigned[x_index:]
    L_sub_active=L_sub[x_index:]
    


    if len(L_sub_active)==0:
        #if len(X_sub_active)>=2:
         #   print('a bug in one point problem')
          #  print('X_sub')
        #print('here')
        #print(X_sub)
        #print(Y_sub)
       
            
        cost_subproblem,L_subproblem=one_point_problem(X_sub_active,Y_sub,y_index)
        cost_subproblem=cost_inactive+cost_subproblem
        
        L_subproblem=L_sub_inactive+L_subproblem
        subproblem_previous=update_problem(cost_subproblem,L_subproblem)
        #print(subproblem_previous)
        return cost_subproblem,L_subproblem,subproblem_previous
  
            
     
    cost_destroy_onepoint={'cost':numpy.inf,'x_index_destroy':[],'cost_previous':0}# we will store the best transporation plan which destroies one point 
    #find the optimal cost and plan that destroy one point 


    
    for l in range(len(X_sub_active)-1,-1,-1): # this is the index we extract x_l,y_l-1
        
        xl1=X_sub_active[l]
        yl=Y_sub_active[l-1]
        #print('x select is'+str(xl1))
        #print('y select is'+str(yl))
        c_xl1yl=cost_function(xl1, yl)
        X_sub_active_left=X_sub_active[0:l]
        X_sub_active_right=X_sub_active[l+1:]
        Y_sub_active_left=Y_sub_active[0:l]
        Y_sub_active_right=Y_sub_active[l:]
        cost_left=sum(cost_function(X_sub_active_left,Y_sub_active_left))+2*Lambda # it used to compute the previous cost and the total cost
        cost_right=sum(cost_function(X_sub_active_right,Y_sub_active_right)) 
        cost_destroy_xl1=cost_left+cost_right
        


        if cost_destroy_xl1<cost_destroy_onepoint['cost']:
            cost_destroy_onepoint['cost']=cost_destroy_xl1
            cost_destroy_onepoint['cost_previous']=cost_left
            cost_destroy_onepoint['x_index_destroy']=l # store the index of the destroyed point x_(l+1)

        #print(cost_destroy_onepoint)
        if c_xl1yl>=2*Lambda:# That is, we will nolonger consider other plan since they will not be optimal. 
            break
    # find the unique plan which preserve all points (if exists)
    cost_destroy_zeropoint=None # we will store the unique best transportation cost which does not destroy points

    if l==0 and y_index>=0:# in this case we need to consider the plan preserve all points
        cost_xkyl=cost_function(X_sub_active[0],Y_sub[y_index])
        if cost_xkyl<2*Lambda:
            cost_destroy_zeropoint=sum(cost_function(X_sub_active[1:],Y_sub_active))+cost_xkyl
    
    # find the optimal cost and plan


    if cost_destroy_zeropoint!=None and cost_destroy_zeropoint<cost_destroy_onepoint['cost']: 
        cost_subproblem=cost_destroy_zeropoint+cost_inactive
        L_subproblem=L_sub_inactive+[y_index]+L_sub_active
        return cost_subproblem,L_subproblem,subproblem_previous
        
        
    else: 
        cost_subproblem=cost_destroy_onepoint['cost']+cost_inactive
        #print(cost_destroy_onepoint)
        x_index_destroy=cost_destroy_onepoint['x_index_destroy']        
        L_subproblem=L_sub_inactive+L_sub_active[0:x_index_destroy]+[-numpy.inf]+L_sub_active[x_index_destroy:]
        L_previous=L_sub_inactive+L_sub_active[0:x_index_destroy]+[-numpy.inf]
        cost_previous=cost_inactive+cost_destroy_onepoint['cost_previous']
        subproblem_previous=update_problem(cost_previous,L_previous)
        return cost_subproblem,L_subproblem,subproblem_previous

    #Get the whole optimal plan             
    


def OPT_1D_v3(X,Y):
    def cost_plan_select():
        nonlocal cost
        nonlocal L
        nonlocal problem_previous
        #nonlocal subproblem_previous
        # select the best cost
        subproblem_previous=None
        cost_book={}
        
        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+2*Lambda  
            elif case=='1n': # incremental cost without conflict
                cost_book[case]=cost+cost_xk_yl
            elif case=='1c' and y_index_last+1<=m-1: # there exists right point
                cost_xk_ylast1=cost_function(xk,Y[y_index_last+1])
                if cost_xk_ylast1<2*Lambda:
                    cost_book[case]=cost+cost_xk_ylast1
            elif case=='2': # cost for recursive problem                
                cost_xk_ylast=cost_function(xk,Y[y_index_last])
                x_startindex,y_startindex=startindex(problem_previous['L_previous'])
                
                if cost_xk_ylast<2*Lambda and y_startindex<=m:
                    X_sub=X[x_startindex:k]
                    Y_sub=Y[y_startindex:y_index_last]
                    L_sub=L[x_startindex:k] # we need the last assigned index since we need to retrieve the closest unassigend y_index

                    
                    
                    index_adjust(L_sub,-y_startindex)
                    cost_subproblem,L_subproblem,subproblem_previous=OPT_subproblem(X_sub,Y_sub,L_sub)
                    cost_book[case]=problem_previous['cost_previous']+cost_subproblem+cost_xk_ylast   
                    index_adjust(L_subproblem,y_startindex)
                    if subproblem_previous!=None:
                        index_adjust(subproblem_previous['L_previous'],y_startindex)
        

        # update transportation plan L            
     
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]
        if min_case=='1n':
            y_index=l # we make the problem smaller and need to modify th index
            L.append(y_index)
        elif min_case=='1c':
            y_index=y_index_last+1
            L.append(y_index)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2':
            L=problem_previous['L_previous']+L_subproblem+[y_index_last]
             
        update_problem(cost,L,problem_previous,min_case,subproblem_previous)
    
            
   


    n=len(X)
    m=len(Y)
 
    L=[] # save the optimal plan
    cost=0 # save the optimal cost
  
    # initial the subproblem
    problem_previous={
        'L_previous':[],
        'cost_previous':0,
        }
    
    for k in range (n):
        xk=X[k]
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively 
        x_startindex,y_startindex=startindex(problem_previous['L_previous'])                
        Y1=Y[y_startindex:]

        if len(Y1)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue

        l,cost_xk_yl=closest_y(xk,Y1)
        l=l+y_startindex
        if cost_xk_yl>=2*Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            continue
        
        y_index_last=L[-1] # index of last y 
        

        if l>y_index_last:# No conflict L[-1]=y_index_last 
            case_set=case_set-{'1c','2'}
        elif l<=y_index_last:# conflict
            case_set=case_set-{'1n'}
            
        cost_plan_select()
#        x_startindex,y_startindex=update_subproblem(L,x_startindex,y_startindex)
        
      
    return cost,L,problem_previous



def POT_1D(X,Y): 
    n=len(X)
    m=len(Y)
 
    L=[] # save the optimal plan
    cost=0 # save the optimal cost
  
    # initial the subproblem
    problem_previous={
        'L_previous':[],
        'cost_previous':0,
        }
    
    for k in range (n):
        xk=X[k]
        l,cost_xk_yl=closest_y(xk,Y)
        
        if len(L)==0 or l>L[-1]:# No conflict, L[-1] is the y_index last assigned
           cost+=cost_xk_yl
           L.append(l)
           continue 
        
        # this is the case for conflict: 
        y_index_last=L[-1]
        # compute the first cost 
        if y_index_last+1<=m-1:
            ylast1=Y[y_index_last+1]
            c_xk_ylast1=cost_function(xk,ylast1)
            cost1=cost+c_xk_ylast1
        else:
            cost1=numpy.inf 
        
        # compute the second cost 
        x_index,y_index=retrieve_unassigned_y(L)
        if y_index>=0:
        
            L1=L[0:x_index]+[y_index]+L[x_index:]

            Y_assigned=Y.take(L1)
            X_assigned=X[0:k+1]
            cost2=sum(cost_function(X_assigned,Y_assigned))
        else:
            cost2=numpy.inf
        if cost1<cost2:
            cost=cost1
            L.append(y_index_last+1)
        elif cost2<=cost1:
            cost=cost2
            L=L1.copy()
 
        
      
    return cost,L




Lambda=7
p=2

for i in range(100):
    X=numpy.random.rand(12)*8
    Y=numpy.random.rand(20)*7+0.5
    X.sort()
    Y.sort()
    
    Cost1,L1=OPT_1D_v1(X,Y)
    Cost2,L2=POT_1D(X,Y)
#    print('L2 is'+str(L2))
 #   print('L1 is'+str(L1))
 #   print('cost difference is'+str(Cost1-Cost2))
    
    if abs(Cost1-Cost2)>0.0001:
        print('there is error in distance')
        print('X is'+str(X))
        print('Y is'+str(Y))
        print(Cost1-Cost2)
        print('L1 is'+str(L1))
        print('L2 is'+str(L2))



#print(L1)

# print(L2)



# X=numpy.array([3.9,4,4.1])
# Y_sub=numpy.array([0,4,4.1])
# L=[1,2]

# X_sub=numpy.array([0.1,0.5])
# Y_sub=numpy.array([0.6])
# L_sub=[0,1]
# cost_subproblem,L_subproblem,previous=OPT_subproblem(X_sub,Y_sub,L_sub)
# print(L_subproblem)
# print(L_subproblem)

# print(previous)

#Y=numpy.array([0.00751822, 0.08950564, 0.48178797, 0.52626412])

# x_startindex=0
# y_startindex=0

# X=numpy.array([1.11071773, 1.80046162, 2.09260222, 3.73466541, 4.99252567])
# #X_sub=numpy.array([1.11071773,1.80046162])

# Y=numpy.array([1.6171406 , 2.04429873, 5.07767902])
# #Y_sub=numpy.array([1.6171406])
# L_sub=[0,1]

# Cost1,L1=OPT_1D_v1(X,Y)
# Cost2,L2=OPT_1D_v2(X,Y)
 
# #X_sub=X[x_startindex:]
#Y_sub=Y[y_startindex:-1]
#L_sub=L[x_startindex:-1]
#C

        
      




##print(L2)   
            


        

