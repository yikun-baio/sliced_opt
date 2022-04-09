
import math
import numpy


class Node():
    def __init__(self, path,value=0):
        self.path =path
        self.value=value
        self.children = []

class Tree():
    def __init__(self,root):
        self.root =root



def create_tree(root, n, d):
    if d == 0:
#       value = 10
#       child = Node(value)
        return None

    for i in range(n):
        value = 10
        child = Node(value)
        create_tree(child, n, d - 1)
        root.children.append(child)
        

    

def print_tree(root, d=0):
    print(f'depth:{d}, value:{root.value}')
    for child in root.children:
        print_tree(child, d=d + 1)
    



def path_convert(path):
    path[0]='root'
    for i in range(1,n+1):
        i_previous=i-1
        while path[i_previous]==0:
            i_previous=i_previous-1
        if path[i]==path[i_previous]:
            path[i]=0

        



def tree_gen(node):
    depth=len(node.path)
    if depth==n+1:
        list_value.append(node.value)
        path=node.path.copy()
        path_convert(path)
        list_path.append(path)
        print(path)
        return None
    x=mu[depth-1] # The depth is also the index of the current x which we need to assign a value for T(x)
    y_previous_index=node.path[-1] #This is the index of y which we assigned in last step
    #print(y_previous_index)
    children_num=n+1-y_previous_index # This is the number of values for T(x_(+1)) we could assign.   
    value_previous=node.value #the value of summation of current points
    for y_index_diff in range(children_num):
        path=node.path.copy() #Copy the current path 
        y_index=y_index_diff+y_previous_index
        path.append(y_index) #Add 
        #print(path)
        y=nu[y_index-1]
        value=value_previous+Dfunction(x,y,y_index_diff)
        child=Node(path,value)
        node.children.append(child)
        tree_gen(child)


def tree_gen1(node):
    depth=len(node.path)
    if depth==n+1:
        list_value.append(node.value)
        list_path.append(node.path)
        print(node.path)
        #print(node.value)
        return None
    x=mu[depth-1] # The depth is also the index of the current x which we need to assign a value for T(x)
    y_previous_index=node.path[-1] #This is the index of y which we assigned in last step
    #print(y_previous_index)
    children_num=n+1-y_previous_index # This is the number of values for T(x_(+1)) we could assign.   
    value_previous=node.value #the value of summation of current T(x)s
    if children_num==1:
        node.value+=Dfunction(0,0,0)*(n+1-depth) # we stop making this tree on this node, so we add all D_bar for remaining xs.
        print(node.value)
        return None
    #print(children_num)
    for y_index_diff in range(children_num):
        path=node.path.copy() #Copy the current path 
        y_index=y_index_diff+y_previous_index
        path.append(y_index) #Add the index of current y which is assigned to T(x) into the path
        #print(path)
        y=nu[y_index-1]
        value=value_previous+Dfunction(x,y,y_index_diff)
        child=Node(path,value)
        node.children.append(child)
        tree_gen1(child)
    
            
def tree_gen2(node):
    depth=len(node.path)
    if depth==n+1:
        return None
    x=mu[depth-1] # The depth is also the index of the current x which we need to assign a value for T(x)
    if depth==1: # initial case
        y_previous_index=0 
    elif depth>=2:
        y_previous_index=node.path[-1] #This is the index of y which we assigned in last step
    #print(y_previous_index)
    children_num=n+1-y_previous_index # This is the number of values for T(x) we could assign.   
    value_previous=node.value #the value of summation of current T(x)s
    if children_num==1:
        return None
    #print(children_num)
    for y_index_diff in range(children_num):
        path=node.path.copy() #Copy the current path 
        y_index=y_index_diff+y_previous_index
        path.append(y_index) 
        print(path)
        y=nu[y_index-1]
        value=value_previous+Dfunction(x,y,y_index_diff)
        child=Node(path,value)
        node.children.append(child)
        tree_gen2(child)
        if y>=x and y_index_diff>0:
            print('here')
            return None # if y_i>=x, it means we no longer need to record the case for T(x)=y_(i+1),y_(i+2),...

Lambda=1/4
mu=[0, 1/4]
nu=[1/2, 1]
n=len(mu)
D_bar=8*Lambda/n

list_value=[]
list_path=[]


node=Node([0],0)
#Tree=Tree(root)   
tree_gen(node)

opt_value=min(list_value)
opt_index=list_value.index(opt_value)
opt_path=list_path[opt_index]
        






#if __name__ == '__main__':
#    print('here')
#    root = Node('1')
#    create_tree(root, n=2, d=3)
#    print_tree(root)
