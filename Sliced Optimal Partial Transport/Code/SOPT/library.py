# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:32:17 2022

@author: laoba
"""

import numpy

def cost_function(x,y,p=2): 
    V=abs(x-y)**p
    return V


def closest_y(x,Y):
    cost_list=cost_function(x,Y)    
    min_index=cost_list.argmin().item()
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
        return [np.array(a) if isinstance(a, list) else a for a in lst]
    else:
        return np.array(lst[0]) if isinstance(lst[0], list) else lst[0]

def list_to_numpy_array(lst):
    r""" Convert a list if in numpy format """
    return numpy.array([a.item() for a in lst])



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
