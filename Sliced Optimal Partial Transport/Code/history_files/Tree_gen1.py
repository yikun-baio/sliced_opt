# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:44:01 2022

@author: laoba
"""


import math
import numpy
from numpy.linalg import norm
from operator import add
#from goto import goto, label

class Node():
    def __init__(self, path,value=0):
        self.path =path
        self.value=value
        self.children = []



class HK():
    def __init__(self, mu, nu,Lambda):
        self.mu=mu
        self.nu=nu
        self.Lambda=Lambda
        self.n=len(self.mu) # the length of mu, which is also the # of points and the depth of the tree
        self.D_bar=8*self.Lambda/self.n
        self.list_value=[]
        self.list_path=[]
        node=Node([0],0)
        self.tree=self.tree_gen1(node)
        self.opt_value=min(self.list_value)
        opt_index=self.list_value.index(self.opt_value)
        self.opt_path=self.list_path[opt_index]


    def Dfunction(self,x,y,transport=0):
        if transport==0:
            return self.D_bar
        elif abs(x-y)/self.Lambda>=math.pi/2:
            return self.D_bar
        elif transport>=1:
            D_xy=8*self.Lambda/self.n*(1-math.cos(min(abs(x-y)/math.sqrt(4*Lambda),math.pi/2)))
            return D_xy

    def path_convert(self,path):
        path[0]='root'
        for i in range(1,self.n+1):
            i_previous=i-1
            while path[i_previous]==0:
                i_previous=i_previous-1
            if path[i]==path[i_previous]:
                path[i]=0
        path=path[1:]
    
    def leaf_store(self,node):
        self.list_value.append(node.value)
        path=node.path.copy()
        self.path_convert(path)
        self.list_path.append(path)
        #print(path)
        #print(node.value)
        
    
    def tree_gen1(self,node):
        depth=len(node.path)
        if depth==self.n+1:
            self.leaf_store(node)
            return None
        x=self.mu[depth-1] # The depth is also the index of the current x which we need to assign a value for T(x)
        y_previous_index=node.path[-1] #This is the index of y which we assigned in last step
        children_num=self.n+1-y_previous_index # This is the number of values for T(x_(+1)) we could assign.   
        value_previous=node.value #the value of summation of current points
        for y_index_diff in range(children_num):
            path=node.path.copy() #Copy the current path 
            y_index=y_index_diff+y_previous_index
            path.append(y_index) #Add 
            #print(path)
            y=nu[y_index-1]
            value=value_previous+self.Dfunction(x,y,y_index_diff)
            child=Node(path,value)
            node.children.append(child)
            self.tree_gen1(child)
    
    def tree_gen2(self,node):
        depth=len(node.path)
        if depth==self.n+1:
            self.leaf_store(node)
            return None
        x=mu[depth-1] # The depth is also the index of the current x which we need to assign a value for T(x)
        y_previous_index=node.path[-1] #This is the index of y which we assigned in last step
        #print(y_previous_index)
        if y_previous_index==self.n:# In this case we no longer need to aign value for remaining T(x)s. 
            n_remaining=self.n+1-depth # the number of remaining xs which is not assigned. 
            path_remain=[self.n]*n_remaining
            node.path=node.path+path_remain
            node.value=node.value+self.Dfunction(0,0,0)*n_remaining
            self.leaf_store(node)
            return None
        
        children_num=self.n+1-y_previous_index # This is the number of values for T(x) we could assign.   
        value_previous=node.value #the value of summation of previous T(x)s
        #print(children_num)
        for y_index_diff in range(children_num):
            path=node.path.copy() #Copy the current path 
            y_index=y_index_diff+y_previous_index
            path.append(y_index) 
            #rint(path)
            y=nu[y_index-1]
            value=value_previous+self.Dfunction(x,y,y_index_diff)
            child=Node(path,value)
            node.children.append(child)
            self.tree_gen2(child)
            
    def tree_gen3(self,node):
        depth=len(node.path)
        if depth==self.n+1:
            self.leaf_store(node)
            return None
        x=mu[depth-1] # The depth is also the index of the current x which we need to assign a value for T(x)
        y_previous_index=node.path[-1] #This is the index of y which we assigned in last step
        #print(y_previous_index)
        if y_previous_index==self.n:# In this case we no longer need to aign value for remaining T(x)s. 
           n_remaining=self.n+1-depth # the number of remaining xs which is not assigned. 
           path_remain=[self.n]*n_remaining
           node.path=node.path+path_remain
           node.value=node.value+self.Dfunction(0,0,0)*n_remaining
           self.leaf_store(node)
           return None
            
        children_num=self.n+1-y_previous_index # This is the number of values for T(x) we could assign.   
        value_previous=node.value #the value of summation of previous T(x)s
        #print(children_num)
        for y_index_diff in range(children_num):
            path=node.path.copy() #Copy the current path 
            y_index=y_index_diff+y_previous_index
            path.append(y_index) 
            #rint(path)
            y=nu[y_index-1]
            value=value_previous+self.Dfunction(x,y,y_index_diff)
            child=Node(path,value)
            node.children.append(child)
            self.tree_gen3(child)
            if y>=x and y_index_diff>=1: # Since if y_j>x_i, we do not consider to assign y_(j+1) to T(x)
                return None



Lambda=1/4
mu=[ 0, 1/4, 2]
nu=[ 0, 1/2, 1]
#A=HK(mu,nu,Lambda)
#print('The optimal value is')
#print(A.opt_value)
#print('The optimal 1-1 mapping is')
#print(A.opt_path)

p=2


def mass_destroy(v):
    if v<2*Lambda:
        return False
    else:
        return True
                    
                







def retrieve_unassigned_y(L):
    y_index_assigned=L[-1] 
    if y_index_assigned<0:
        print('a bug')
        return None
    for y_index in range(y_index_assigned,0,-1):
        if y_index not in L:
            x_index_last_assigned=len(L)
            x_index=x_index_last_assigned-(y_index_assigned-y_index)
            return x_index,y_index
    return 0,-1





def OPT_1D_empty_Y(self,X):
    n=len(X)
    L=[-numpy.inf]*n
    cost=2*Lambda*n
    return cost,L
    


        
        

def OPT_1D_v4(X,Y):
    def cost_plan_select():
        nonlocal cost
        nonlocal L   
        # select the best cost
        cost_book={}

        for case in case_set:
            if case=='0': # cost for destroy mass 
                cost_book[case]=cost+2*Lambda  
            elif case=='1n': # incremental cost without conflict
      #          if k==3:
 #                   print('here')
                cost_book[case]=cost+cost_xk_yl
            elif case=='1c' and index_y_last+1<=m-1: # there exists right point
                cost_xk_ylast1=cost_function(xk,Y[index_y_last+1])
                if cost_xk_ylast1<2*Lambda:
                    cost_book[case]=cost+cost_xk_ylast1
            elif case=='2': # cost for recursive problem
            #print('conflict')
                cost_xk_ylast=cost_function(xk,Y[index_y_last])
                if cost_xk_ylast<2*Lambda and sub_problem_start_y<=m:
                    X_sub=X[sub_problem_start_x:k]
                    Y_sub=Y[sub_problem_start_y:index_y_last]
                    L_sub=L[sub_probelem_start_x:]
                    #X_sub=X[0:k]
                    #Y_sub=Y[0:index_y_last]
                    cost_subproblem,L_subproblem=OPT_subroblem(X_sub,Y_sub)   
                    #  print('the start index'+str(sub_problem_start_y))
                    cost_book[case]=cost_previous+cost_subproblem+cost_xk_ylast
                    L_subproblem_adjust=[index+sub_problem_start_y for index in L_subproblem] # adjust the index of the solution of subproblem
                

                
                if cost_xk_ylast<2*Lambda and sub_problem_start_y<=m:
                    #X_sub=X[sub_problem_start_x:k]
                    Y_sub=Y[sub_problem_start_y:index_y_last]
                    #X_sub=X[0:k]
                    #Y_sub=Y[0:index_y_last]
                    cost_subproblem,L_subproblem=OPT_sub(X,Y,)   
                  #  print('the start index'+str(sub_problem_start_y))
                    cost_book[case]=cost_previous+cost_subproblem+cost_xk_ylast
                    L_subproblem_adjust=[index+sub_problem_start_y for index in L_subproblem] # adjust the index of the solution of subproblem

        
        # Update optimal plan L 
        min_case=min(cost_book,key=cost_book.get)
        cost=cost_book[min_case]

     

        if min_case=='1n':
            y_index=l # we make the problem smaller and need to modify th index
            L.append(y_index)
        elif min_case=='1c':
            y_index=index_y_last+1
            L.append(y_index)
        elif min_case=='0':
            L.append(-numpy.inf)
        elif min_case=='2':
            L=L_previous+L_subproblem_adjust+[index_y_last]
            
    def update_subproblem():
        nonlocal Y1
        nonlocal cost_previous
        nonlocal L_previous
        nonlocal sub_problem_start_y
        nonlocal sub_problem_start_x
        
        # update the subproblem
        index_y_current=L[-1]
        L_assigned=[i for i in L if i>=0]
        if index_y_current<=-1 and len(L_assigned)>=1: # x_k is destroyed in this step

            index_y_last_aligned=max(L_assigned) # index of last aligned y 
            sub_problem_start_x=k+1 # 
            sub_problem_start_y=index_y_last_aligned+1 #
           # print(sub_problem_start_y)
            Y1=Y[sub_problem_start_y:]
            cost_previous=cost
            L_previous=L.copy()
            #print('previous plan here is'+str(L_previous))


    n=len(X)
    m=len(Y)
    X_indexes=list(range(0,n))
    Y_indexes=list(range(0,m))

    
    L=[] # save the optimal plan
    cost=0 # save the optimal cost
  
    # save the subproblem 
    sub_problem_start_x=0 
    sub_problem_start_y=0 
    cost_previous=0 
    L_previous=[] 
    Y1=Y.copy()
    Y1_indexes=Y_indexes.copy()
    for k in X_index:
        case_set={'0','1n','1c','2'} # 0 is for cost without conflict, 1 is for cost with conflict, 2 is cost with destroying, 3 is cost we solve the problem recursively
        

        if len(Y1)==0: # There is no y, so we destroy point 
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            update_subproblem()
            continue

        l,cost_xk_yl=closest_y(xk,Y1,Y1_indexes)
        l=l+sub_problem_start_y
        if cost_xk_yl>=2*Lambda: # closest distance is 2Lambda, then we destroy the point
            case_set=case_set-{'1n','1c','2'}
            cost_plan_select()
            update_subproblem()
            continue
        
        if len(L)==0:# No conflict
            case_set=case_set-{'1c','2'}
            cost_plan_select()
            update_subproblem()
            continue
        
        index_y_last=L[-1] # index of last y 
#        print('Last index is'+str(index_y_last))
  
       # here index_y_last_aligned=index_y_last
        if l>index_y_last:# No conflict L[-1]=index_y_last 
            case_set=case_set-{'1c','2'}
        elif l<=index_y_last:# conflict
            case_set=case_set-{'1n'}
        cost_plan_select()
        update_subproblem()
        
       
    return cost,L 



def OPT_subproblem(X_sub,Y_sub,L_sub,sub_problem_start_x=0,sub_problem_start_y=0):
    x_index,y_index=retrieve_unassigned_y(L_sub)
    print(x_index)
    # compute the unchanged cost
    m_sub=len(Y_sub)
    n_sub=len(X_sub)
    L1=[] # This is the plan without point destroy 
    if x_index>0:
        L_previous=L_sub[0:x_index]
        X_previous=X[0:x_index]
        Y_previous=Y[0:x_index]
        previous_cost=sum(cost_function(X_previous,Y_previous))
        print('previous cost is'+str(previous_cost))
        for l in range(m_sub-1,x_index-1,-1):
            xl=X_sub[l]
            yl=Y_sub[l]
            c_xlyl=cost_function(xl, yl)
            L1.insert(0,l+x_index-1)
            if c_xlyl>=2*Lambda:
                break
            
        print(L_previous+L1)
            
    return L1
            





Lambda=0.5

for i in range(1):
    X=numpy.random.rand(5)*5
    Y=numpy.random.rand(6)*8
    X.sort()
    Y.sort()
    
    Cost1,L1=OPT_1D_v1(X,Y)
    Cost2,L2=OPT_1D_v3(X,Y)
    #print('cost difference is'+str(Cost1-Cost2))
    if abs(Cost1-Cost2)>0:
        #print(Cost1-Cost2)
        print(L2)
        print(L1)
        print('there is a consistent!')
        

X=numpy.array([0, 1, 4,4.1])
Y=numpy.array([0, 1, 3.9,4,4.1])

Cost,L=OPT_1D_v2(X,Y)

X_sub=numpy.array([0, 1, 4,4.1])
Y_sub=Y[:-1]
OPT_subproblem(X_sub,Y_sub,L)

##print(L2)    
    
