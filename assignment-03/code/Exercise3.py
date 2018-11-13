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

plt.xlabel('$\eta$',font_size=40)
plt.ylabel('$r_n$' )

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.legend()