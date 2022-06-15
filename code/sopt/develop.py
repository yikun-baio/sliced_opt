# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:31:18 2022

@author: laoba
"""
import numpy as np
import torch
from library import *
from sliced_opt import random_projections
import torch.multiprocessing as mp
from numba import vectorize,jit,guvectorize

class sopt():
    def initializer(self,init_data):
        aux_data = init_data
    
    def __init__(self,X,Y,Lambda,n_projections=6,Type=None):
        self.device=X.device.type
        self.dtype=X.dtype
        self.n,self.d=X.shape
        self.m=Y.shape[0]
        self.n_projections=n_projections
        self.Lambda=Lambda
        self.Type=Type
        self.X=X
        self.Y=Y
        projections=random_projections(self.d,self.n_projections,self.device,self.dtype)
        self.get_projections(self.n_projections)
        self.get_plans()
        projections=projections.to(self.device)
    

    def sliced_cost(self):
        mass=torch.sum(self.plans>=0)/self.n_projections
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.plans)
        return cost,mass

    def get_projections(self,projections):
        projections=random_projections(self.d,self.n_projections,self.device,self.dtype)
        self.X_sliced=torch.matmul(projections.T,self.X.T)
        self.Y_sliced=torch.matmul(projections.T,self.Y.T)
    
    def get_plans(self):
        X=self.X
        Y=self.Y
        X_sliced=self.X_sliced # remove gradient 
        Y_sliced=self.Y_sliced # remove gradient 
        self.X_sliced=self.X_sliced.detach().cpu()
        self.Y_sliced=self.Y_sliced.detach().cpu()
        self.X=self.X.detach().cpu()
        self.Y=self.Y.detach().cpu()
        self.plans=torch.zeros([self.n_projections,self.n],dtype=int)
        self.costs=torch.zeros([self.n_projections])
        n_cpu=mp.cpu_count()
        size=int(self.n_projections/n_cpu/2)
        pool=mp.Pool()
        r=pool.map_async(self.one_slice, range(self.n_projections),chunksize =8) 

        pool.close()
        pool.join()
        self.X_sliced=X_sliced
        self.Y_sliced=Y_sliced
        self.X=X
        self.Y=Y
    
    
   # @vectorize(['int32(int32)'], target='cuda')
    def one_slice(self,i):
        X_theta=self.X_sliced[i]
        Y_theta=self.Y_sliced[i]
        X_theta_s,indices_X=X_theta.sort()
        Y_theta_s,indices_Y=Y_theta.sort()
        cost1,L1=opt_1d_T(X_theta_s, Y_theta_s, self.Lambda)
        L2=recover_indice(indices_X,indices_Y,L1)
        self.plans[i,:]=L2
        self.costs[i]=cost1
    
    def refined_cost(self,Xs,Ys,plans):
        N=Xs.shape[0]
        Lx=[torch.arange(self.n)[plans[i]>=0] for i in range(N)]
        Ly=[plans[i][plans[i]>=0] for i in range(N)]
        X_take=torch.cat([Xs[i][Lx[i]] for i in range(N)])
        Y_take=torch.cat([Ys[i][Ly[i]] for i in range(N)])        
        cost_trans=torch.sum(cost_function_T(X_take, Y_take))
        destroy_mass=N*self.n-X_take.shape[0]
        penulty=2*self.Lambda*destroy_mass
        return (cost_trans+penulty)/N




        

class max_sopt(sopt):
    
    def max_cost(self):
        max_index=self.costs.argmax()
        max_plan=self.plans[max_index].reshape([1,self.n])
        X_max=self.X_sliced[max_index].reshape([1,self.n])
        Y_max=self.Y_sliced[max_index].reshape([1,self.m])
        max_cost=self.refined_cost(X_max, Y_max, max_plan)
        max_mass=torch.sum(max_plan>=0)        
        return max_cost,max_mass
    

class sopt_majority(sopt):
    def __init__(self,X,Y,Lambda,n_projections=2,Type=None,N_destroy=0):
        self.sopt.__init__(X,Y,Lambda,n_projections,Type)
        #self.n_preserve=N
        self.new_plan(N)
    
    def new_plan(self,N):
        self.new_plans=self.plans.clone()
        X_frequency=torch.sum(self.plans>=0,0)
        X_lowest_frequency=X_frequency.sort().indices[0:N_destroy]
        self.new_plans[:,X_lowest_frequency]=-1
    def sliced_cost(self):
        mass=torch.sum(self.plans>=0)/self.n_projections
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.new_plans)
        return cost,mass
    
class sopt_delta(sopt):
    def __init__(self,X,Y,Lambda,n_projections=2,Type=None,delta=0):
        self.sopt.__init__(X,Y,Lambda,n_projections,Type)
        #self.n_preserve=N
        self.new_plan(delta)
    
    def new_plan(self,delta):
        self.new_plans=self.plans.clone()
        X_frequency=torch.sum(self.plans>=0,0)
        X_lowest_frequency=torch.array(self.n)[X_frequency<=delta]
        self.new_plans[:,X_lowest_frequency]=-1
    def sliced_cost(self):
        mass=torch.sum(self.plans>=0)/self.n_projections
        cost=self.refined_cost(self.X_sliced,self.Y_sliced,self.new_plans)
        return cost,mass
    
    
    





def opt_sub(X1,Y1,L1,Lambda):
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
        if -1 in L_sub:
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
        cost_left=sum(cost_function(X1_act_left,Y1_act_left))+Lambda # it is used to compute the previous cost and the total cost
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
        L_sub=L1_inact+L1_act[0:i_destroy]+[-1]+L1_act[i_destroy:]
        L_sub_pre=L1_inact+L1_act[0:i_destroy]+[-1]
        cost_sub_pre=cost_inact+cost_d1_opt['cost_pre']
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    else: 
        cost_sub=cost_d0+cost_inact
        L_sub=L1_inact+[j_act]+L1_act
        return cost_sub,L_sub,cost_sub_pre,L_sub_pre
    
        

#@guvectorize(['(float32[:],float32[:],float32)'],'(n),(m),(i)->(n)',target='cuda')
def text(X,Y,Lambda):
    return X[:]+Y[:]+Lambda


def opt_1d_v2(X,Y,Lambda):
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
            L.append(-1)
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
        i_start,j_start=startindex_np(L_pre)                
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
        
      
    return L
    

    
    
    
    

        
        

    

    


    
    
    
    
        
    

    
    

