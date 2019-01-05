# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:04:30 2018

@author: Mark
"""

def k(x,x2,T):
    T[0]*np.exp(-(T[1]/2)*np.power(np.abs(x-x2),2))+T[2]+T[3]*x*x2
    
def GramMatrix()    
    
#%%
    
