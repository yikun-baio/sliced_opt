#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 20:34:01 2022

@author: baly
"""


import matplotlib.pyplot as plt
import numpy as np
import torch

import pandas as pd




List=torch.load('costlist.pt')
cost_list=List['cost_list']
coste_list=List['coste_list']
labels=["0","1", "2", "3","4","5","6","7","8","9","O",'I',"II","III","IV","V","VI","VII","VIII","IX"]
df = pd.DataFrame(cost_list).T
df.to_excel(excel_writer = "test.xlsx")
df = pd.DataFrame(coste_list).T
df.to_excel(excel_writer = "teste.xlsx")

# fig, ax =plt.subplots(1,1,figsize=(20,20))

# column_labels=labels
# df=pd.DataFrame(cost_list,columns=column_labels)
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=df.values,
#         colLabels=df.columns,
#         rowLabels=labels,
#         rowColours =["yellow"] * 20,  
#         colColours =["yellow"] * 20,
#         loc="center")

plt.show()
