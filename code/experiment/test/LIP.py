#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:20:02 2022

@author: baly
"""

import numpy as np
from ortools.linear_solver import pywraplp

def cost_matrix(vector_list):
    N=len(vector_list) #[0]
    cost_M=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            vi=vector_list[i]
            vj=vector_list[j]
            cost_M[i,j]=np.dot(vi,vj)/(np.linalg.norm(vi)*np.linalg.norm(vj))
    return cost_M

def solution_matrix(solution):
    N=len(solution)
    solution_M=[]
    for i in range(N):
        array=[]
        for j in range(N):
            xi=solution[i]
            xj=solution[j]
            array.append(xi+xj-1)
        solution_M.append(array)
    return solution_M

vector_list=[np.array([1,0]),np.array([0,1]),np.array([1,1])]
N=len(vector_list)
K=2
    
solver = pywraplp.Solver.CreateSolver('SCIP')
infinity = solver.infinity()
solution=[]
for i in range(N):
    xi=solver.IntVar(0.0, 1.0, 'x'+str(i))
    solution.append(xi)


print('Number of variables =', solver.NumVariables())
solver.Add(np.sum(solution) ==K)

cost_M=cost_matrix(vector_list)
solution_M=solution_matrix(solution)

object_f=np.sum(solution_M*cost_M)
solver.Minimize(object_f)
status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    for i in range(N):
        print('x'+str(i)+' =', solution[i].solution_value())

else:
    print('The problem does not have an optimal solution.')