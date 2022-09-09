#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:42:15 2022

@author: baly
"""

import torch
import numpy as np 
import os
import sys
import ot
import matplotlib.pyplot as plt
import time

work_path=os.path.dirname(__file__)
print('work_path is', work_path)
loc1=work_path.find('/code')
parent_path=work_path[0:loc1+5]
sys.path.append(parent_path)


lab_path=parent_path
os.chdir(lab_path)
sys.path.append(parent_path)

from sopt3.opt import *
from sopt3.library import *
from sopt3.lib_ot import *
start_n=50
end_n=1000
step=5
Lambda_list= np.array([10.0])
n_list=np.array(range(start_n,end_n,step))
for Lambda in Lambda_list:
    start_n=50
    end_n=1000
    step=5
    
    cost_list=torch.load('experiment/test/results/accuracy_list'+str(Lambda)+'.pt')
    cost_v2_list=cost_list['cost_v2_list']
    cost_v2_a_list=cost_list['cost_v2_a_list'] 
    cost_pr_list=cost_list['cost_pr_list']
    cost_lp_list=cost_list['cost_lp_list']
    
    cost_v2_list_n=cost_v2_list/n_list
    cost_v2_a_list_n=cost_v2_a_list/n_list 
    cost_pr_list_n=cost_pr_list/n_list
    cost_lp_list_n=cost_lp_list/n_list
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(start_n,end_n,step),cost_v2_list,'-',label='ours')
    plt.plot(range(start_n,end_n,step),cost_v2_a_list,label='ours-apro')
    plt.plot(range(start_n,end_n,step),cost_pr_list,label='Lp (primal): python OT')
    plt.plot(range(start_n,end_n,step),cost_lp_list,label='Lp: python OT')
    #lt.plot(range(start_n,end_n,step),cost_lp_list,label='Sinkhorn: python OT')
    plt.xlabel("n: size of X")
    plt.ylabel("OPT distance")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.2),
              fancybox=True, shadow=True, ncol=3)
#    plt.savefig("myImage.png", format="png", dpi=resolution_value)
    plt.savefig('experiment/test/results/accuracy'+str(Lambda)+'.png',format="png",dpi=2000,bbox_inches='tight')
    plt.show()
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.semilogy(range(start_n,end_n,step),cost_v2_list-cost_lp_list,'-',label='error |outs-lp|')
    plt.semilogy(range(start_n,end_n,step),cost_v2_a_list-cost_lp_list,label='error |ours_a-lp|')
    plt.xlabel("n: size of X")
    plt.ylabel("error")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.13),
              fancybox=True, shadow=True, ncol=3)
    plt.savefig('experiment/test/results/accuracy_error'+str(Lambda)+'.png',format="png",dpi=2000,bbox_inches='tight')
    plt.show()
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(n_list,cost_v2_list_n,'-',label='ours')
    plt.plot(n_list,cost_v2_a_list_n,label='ours-apro')
    plt.plot(n_list,cost_pr_list_n,label='Lp (primal): python OT')
    plt.plot(n_list,cost_lp_list_n,label='Lp: python OT')
    #lt.plot(range(start_n,end_n,step),cost_lp_list,label='Sinkhorn: python OT')
    plt.xlabel("n: size of X")
    plt.ylabel("nomalized OPT distance")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.2),
              fancybox=True, shadow=True, ncol=3)
#    plt.savefig("myImage.png", format="png", dpi=resolution_value)
    plt.savefig('experiment/test/results/accuracy'+str(Lambda)+'.png',format="png",dpi=2000,bbox_inches='tight')
    plt.show()
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(n_list,cost_v2_list_n-cost_lp_list_n,'-',label='error |ours-lp|')
    plt.plot(n_list,cost_v2_a_list_n-cost_lp_list_n,label='error |ours_a-lp|')
    plt.xlabel("n: size of X")
    plt.ylabel("error")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.13),
              fancybox=True, shadow=True, ncol=3)
    plt.savefig('experiment/test/results/accuracy_error'+str(Lambda)+'.png',format="png",dpi=2000,bbox_inches='tight')
    plt.show()