# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:29:57 2018

@author: mbeijer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#%%

plt.figure("32b")
x = np.linspace(-2,2,10000)

y1 = np.abs(1-x)
y2 = np.abs(1-2*x)

plt.plot(x,y1,label='$\lambda_1$')
plt.plot(x,y2,label='$\lambda_2$')

plt.xlabel('$\eta$',size=40)
plt.ylabel('$r_n$',size=40)




plt.legend(prop={'size': 20})

#%%


def g(x,y,a1,a2,lambda1,lambda2):
    return (lambda1/2)*(x-a1)**2+ (lambda2/2)*(y-a2)**2


def GradDesc(x,y,a1,a2,lambda1,lambda2,eta):
    xnew = x + eta*lambda1*(x - a1)
    ynew = y + eta*lambda1*(y - a2)
    return xnew,ynew


def plotContour(xmin,xmax,ymin,ymax,xamount,yamount,a1,a2,lambda1,lambda2,name,ax,fig):
    x = np.linspace(xmin,xmax,xamount)
    y = np.linspace(ymin,ymax,yamount)
    x,y = np.meshgrid(x,y)
    z = g(x,y,a1,a2,lambda1,lambda2)
    plt.xlabel('x',fontsize=20,labelpad=20)
    plt.ylabel('y',fontsize=20,labelpad=20)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return ax,fig

def plotTraject(Times,x,y,a1,a2,lambda1,lambda2,eta,ax):
    for i in range(Times):
        xnew,ynew = GradDesc(x,y,a1,a2,lambda1,lambda2,eta)
        X = np.array([x,xnew])
        Y = np.array([y,ynew])
        Z = g(X,Y,a2,a2,lambda1,lambda2)+2
        ax.plot(X, Y, Z, color='orange')
        x,y = xnew,ynew
        print(x,y)


#%%
    
xmin = -4
xmax = 4
ymin = -4
ymax = 4
xamount = 1000
yamount = 1000
a1 = 1
a2 = 1
lambda1 = 1
lambda2 = 1
fig = plt.figure("Test")
ax = fig.gca(projection='3d')
Times = 100

x,y = 2,2
eta = 10e-3

ax,fig = plotContour(xmin,xmax,ymin,ymax,xamount,yamount,a1,a2,lambda1,lambda2,"test",ax,fig)


plotTraject(Times,x,y,a1,a2,lambda1,lambda2,eta,ax)

