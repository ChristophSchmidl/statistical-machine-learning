# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:29:57 2018

@author: mbeijer
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

x = np.linspace(-2,2,10000)

y1 = np.abs(1-x)
y2 = np.abs(1-2*x)

plt.plot(x,y1,label='$\lambda_1$')
plt.plot(x,y2,label='$\lambda_2$')

plt.xlabel('$\eta$',size=40)
plt.ylabel('$r_n$',size=40)




plt.legend(prop={'size': 20})