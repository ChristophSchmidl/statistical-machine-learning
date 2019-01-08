# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:04:30 2018

@author: Mark
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 50,
         'axes.titlesize':'x-large',
         'xtick.labelsize': 25,
         'ytick.labelsize': 25}
pylab.rcParams.update(params)

#%%


def k(x,x2,T):
    return T[0]*np.exp(-(T[1]/2)*np.power(np.linalg.norm(x-x2),2))+T[2]+T[3]*np.dot(x,x2)

def GramMatrix(X,T):
    N = len(X)
    Gram = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Gram[i][j] = k(X[i],X[j],T)
    return Gram
            
def SemiDef(Matrix):
    FixedMatrix = Matrix + 1e-10*np.identity(len(Matrix))
    EigVals = np.linalg.eigvals(FixedMatrix)
    Lowest = np.min(EigVals)
    IsSemi = np.real(Lowest)>=0
    return IsSemi

def Normal(x,mu,sigma):
    return (1/(np.sqrt(2*np.pi*np.square(sigma))))*np.exp(-np.square(x-mu)/(2*np.square(sigma)))

def MeanSigma(X,xnew,C,t,Theta):
    N = len(x)
    K = np.zeros(N)
    for i in range(N):
        K[i] = k(x[i],xnew,Theta)
    Cinv = np.linalg.inv(C)
    Mean = np.dot(K.transpose(),np.dot(Cinv,t))[0]
    c = k(xnew,xnew,Theta)+1
    Sigma = c - np.dot(K.transpose(),np.dot(Cinv,K))
    return Mean,Sigma
    
#%%Constants
    
Theta = np.array([1,1,1,1])
X = np.linspace(-1,1,101)

Gram = GramMatrix(X,Theta)

print(SemiDef(Gram))

#%%


Norm = Normal(X,0,Gram)
Exp = np.exp(X)

#%%

Theta = np.array([[1, 4, 0, 0], [9, 4, 0, 0], [1, 64, 0, 0], [1, 0.25, 0, 0],
                  [1, 4, 10, 0], [1, 4, 0, 5]])
for j in range(len(Theta)):
    T = Theta[j]
    K = GramMatrix(X,T) 
    for i in range(100000):
        Gaus = np.random.multivariate_normal([0]*101,K)
        Ax = plt.subplot(2, 3, j + 1)
        Ax.set_xticks(np.round(np.linspace(-1, 1, 5), 2))
        for tick in Ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in Ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        Ax.set_xlim([-1,1])
        Ax.plot(X,Gaus)
        Ax.set_title(str(T),size=30)

#%%
x = y = np.linspace(-1,1,21)
x,y = np.meshgrid(x,y)
X = []
for i in range(len(x)):
    for j in range(len(x[0])):
        X.append(np.array([x[i][j],y[i][j]]))
X = np.array(X)


Gram2 = GramMatrix(X,T)

#%%

Theta = np.array([[1, 1, 1, 1], [1, 10, 1, 1], [1, 1, 1, 10]])
for T in Theta:
    fig = plt.figure(str(T))
    fig.suptitle(str(T),fontsize=50)
    Gram2 = GramMatrix(X,T)
    for i in range(4):     
        Z = np.random.multivariate_normal([0]*441,Gram2)
        Z = np.reshape(Z,(21,21))
        ax = fig.add_subplot(221+i,projection='3d')
        ax.set_xticks(np.round(np.linspace(-1, 1, 5), 2))
        ax.set_yticks(np.round(np.linspace(-1, 1, 5), 2))
        surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,linewidth=1, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5,pad=-0.07)
        ax.view_init(azim=-120,elev = 70)
    pylab.get_current_fig_manager().window.showMaximized()
    plt.subplots_adjust(wspace=0, hspace=0)


#%%
    
Theta = np.array([1,1,1,1])
D = np.array([[-0.5,0.5],[0.2,-1],[0.3,3],[-0.1,-2.5]])
x,t = np.hsplit(D,2)
Gram = GramMatrix(x,Theta)
C = Gram + np.identity(len(Gram))
print(MeanSigma(x,0,C,t,Theta))
#%%

plt.scatter(x,t)

X = np.linspace(np.min(x),np.max(x),100)
MeanSig = []

for xpart in X:
    MeanSig.append(MeanSigma(x,xpart,C,t,Theta))

MeanSig = np.array(MeanSig)
Mean,Sigma = np.hsplit(MeanSig,2)

Y1 = Mean
Y2 = Mean+np.sqrt(Sigma)
Y3 = Mean-np.sqrt(Sigma)


plt.plot(X,Y1)
plt.plot(X,Y2,'--')
plt.plot(X,Y3,'--')

#%%

print(MeanSigma(x,-1e50,C,t,Theta))

#%%


Theta2 = np.array([100,100,0,0])

print(MeanSigma(x,-1e50,C,t,Theta2))
print(MeanSigma(x,1e50,C,t,Theta2))
