"""
Created on Sun Jun 26 14:25:29 2022
@author: Soheil Kolouri soheil.kolouri@vanderbilt.edu
"""

import numpy as np
import pot1d 
import matplotlib.pyplot as plt

Lambdas=[1,10,100,1000]
np.random.seed(70)
X1=(np.random.rand(8)-.5)*40
Y1=(np.random.rand(16)-.5)*80
X1.sort()
Y1.sort()
fig,ax=plt.subplots(4,1,figsize=(10,8))
ax[-1].xaxis.set_tick_params(labelsize=16)
for k,Lambda in enumerate(Lambdas):
    out=pot1d.solve(X1,Y1,Lambda)
    ind1=out[3]
    
    ax[k].plot([-40.,40.],[0.5,0.5],'--',c='gray',zorder=0)
    ax[k].plot([-40.,40.],[0.,0.],'--',c='gray',zorder=0)
    ax[k].scatter(X1,np.zeros_like(X1),s=100,marker='o')
    ax[k].scatter(Y1,.5*np.ones_like(Y1),s=100,marker='o')
    for i in range(X1.shape[0]):
        if ind1[i]!=-1:
            ax[k].plot([X1[i],Y1[ind1[i]]],[0.04,.46],'k')
    ax[k].set_ylim(-.1,.6)
    ax[k].set_yticks([])
    if k!=3:
        ax[k].set_xticks([])
    # ax[k].set_xticklabels(fontsize=16)
    ax[k].set_ylabel(r'$\lambda=$%d'%(Lambda),fontsize=18)
fig.savefig('./Lambda.png',bbox_inches='tight')
print('Code successfully executed! The results are saved under ./Lamda.pdf')