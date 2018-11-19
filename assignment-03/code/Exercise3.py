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
    xnew = x - eta*lambda1*(x - a1)
    ynew = y - eta*lambda2*(y - a2)
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
    return ax,fig,np.max(z)

def plotTraject(Times,x,y,a1,a2,lambda1,lambda2,eta,ax):
    printed = False
    for i in range(Times):
        xnew,ynew = GradDesc(x,y,a1,a2,lambda1,lambda2,eta)
        X = np.array([x,xnew])
        Y = np.array([y,ynew])
        Z = g(X,Y,a2,a2,lambda1,lambda2)
        ax.plot(X, Y, Z, color='orange')
        if(not printed and xnew == a1 and ynew == a2):
            print("Converged at step " + str(i+1))
            printed = True
        x,y = xnew,ynew
        #print(x,y)
    print(xnew)
    print(ynew)

def PlotLine(ax,x,y,minval,maxval):
    ax.plot([x,x],[y,y],[minval,maxval],color='red')

#%%
    
    

    

xmin = -100
xmax = 100
ymin = -100
ymax = 100
xamount = 1000
yamount = 1000
a1 = 1
a2 = 1
lambda1 = 1
lambda2 = 0.5
fig = plt.figure("Test")
ax = fig.gca(projection='3d')
Times = 5

x,y = 3,4

eta = 3
#eta = 2/(lambda1+lambda2)

ax,fig,maxval = plotContour(xmin,xmax,ymin,ymax,xamount,yamount,a1,a2,lambda1,lambda2,"test",ax,fig)


plotTraject(Times,x,y,a1,a2,lambda1,lambda2,eta,ax)

PlotLine(ax,a1,a2,0,maxval)
SMALL_SIZE = 40
MEDIUM_SIZE = 50

ax.view_init(elev=57., azim=-106)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize