# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:43:40 2018

@author: Mark
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from mpl_toolkits.mplot3d import Axes3D
import os

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 50,
         'axes.titlesize':'x-large',
         'xtick.labelsize': 25,
         'ytick.labelsize': 25}
pylab.rcParams.update(params)
#%% Functions


def Gaus(X):
    FirstPart = 1/np.power(2*np.pi,len(X)/2)*1/(np.power(np.linalg.det(Sigma),0.5))
    Exp = -(1/2)*np.matmul(np.transpose(X-Mu),np.matmul(np.linalg.inv(Sigma),X-Mu))
    return FirstPart*np.exp(Exp)

def Plot(X,Y,Name,save=False,savename=None,title=None):
    x,y = np.hsplit(X,2)
    x = np.reshape(x,(41,41))
    y = np.reshape(y,(41,41))
    z = np.reshape(Y,(41,41))
    fig = plt.figure(Name,figsize=(32,24))
    ax = fig.gca(projection='3d')
    if(title != None):
        plt.title(title,fontsize=40)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=1, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5,pad=-0.07)
    ax.view_init(azim=-120,elev = 70)
    ax.set_zlabel('\nY  ',fontsize=50)
    ax.set_ylabel('\n$X_1$  ',fontsize=50)
    ax.set_xlabel('\n\n$X_0$',fontsize=50)
    
    if(save):
        pylab.get_current_fig_manager().window.showMaximized()
        ax.set_xlim3d([-2, 2])
        ax.set_ylim3d([-2, 2])
        ax.set_zlim3d([0, 1.5])
        plt.savefig(savename,dpi=100,layout='tight_layout')
        plt.close(Name)
        return
    plt.show()



class NeuralNW():
    def __init__(self,X,Y,eta=0.1,D=2,K=1,M=8):
        self.w1 = np.random.random_sample((D+1,M))-0.5
        self.w2 = np.random.random_sample(M+1)-0.5
        self.X = X
        self.Y = Y
        self.TrainX = []
        self.eta = eta
        for x in self.X:
            self.TrainX.append(np.array([1,x[0],x[1]]))
        self.TrainX = np.array(self.TrainX)


    def setWeights(self,w1,w2):
        self.w1 = w1
        self.w2 = w2


    def setNetworkOnce(self,Num):
        self.HiddenLayer = np.tanh(np.dot(self.w1[0], self.TrainX[Num][0]) + np.dot(self.w1[1] , self.TrainX[Num][1]) + np.dot(self.w1[2],self.TrainX[Num][2]))
        self.HiddenLayer = np.insert(self.HiddenLayer,0,1)
        self.Output = np.sum(self.HiddenLayer*self.w2)

    def TrainNetwork(self,Num):
        self.setNetworkOnce(Num)
        Delta2 = self.Output - self.Y[Num]
        Delta1 = (1-np.square(self.HiddenLayer))*self.w2*Delta2
        Der2 = Delta2*self.HiddenLayer
        Delta1 = Delta1[1:]
        Der1 = np.array([Delta1*self.TrainX[Num][0],Delta1*self.TrainX[Num][1] , Delta1*self.TrainX[Num][2]])
        self.w2 -= self.eta*Der2
        self.w1 -= self.eta*Der1
        
    def TrainNetworkAll(self):
        for i in range(len(self.X)):
            self.TrainNetwork(i)
            
    def TrainNetworkRand(self):
        for i in np.random.permutation(len(self.X)):
            self.TrainNetwork(i)
    
    def Distance(self):
        Out = []
        for i in range(len(self.X)):
            self.setNetworkOnce(i)
            Out.append(self.Output)
        Out = np.array(Out)
        return np.sum(np.abs(self.Y - Out))
        
    def getOutput(self):
        Out = []
        for i in range(len(X)):
            self.setNetworkOnce(i)
            Out.append(self.Output)
        return np.array(Out)
        
        
            
    def PlotOutput(self,Name,save=False,savename=None,title=None):
        Out = []
        for i in range(len(self.X)):
            self.setNetworkOnce(i)
            Out.append(self.Output)
        Out = np.array(Out)
        Plot(self.X,Out,Name,save=save,savename=savename,title=title)
#%%



#%%Constants

Sigma = (2/5)*np.identity(2)
Mu = np.zeros(2)
x = y = np.arange(-2,2+0.1,0.1)
x,y = np.meshgrid(x,y)
X = []
for i in range(len(x)):
    for j in range(len(x[0])):
        X.append(np.array([x[i][j],y[i][j]]))
X = np.array(X)
Y = np.array([Gaus(x) for x in X])
import sys
sys.exit("Stop to make sure the correct path is loaded")
#%%2.1

Plot(X,Y,"Plot Gaussian")






#%% 2.2 + 2.3


Network = NeuralNW(X,Y)
Now = time()

Network.PlotOutput("Initial_Output")

for i in range(2000):
    if (i == 199):
        Network.PlotOutput("Plot_200_iterations")
    Network.TrainNetworkAll()

Network.PlotOutput("Plot_2000_iterations")

#%%

NetworkRand = NeuralNW(X,Y)
for i in range(20):
    NetworkRand.TrainNetworkRand()
print("Error for the random training = ",NetworkRand.Distance())

NetworkRand.PlotOutput("Random_Training")

NetworkNorm = NeuralNW(X,Y)
for i in range(20):
    NetworkNorm.TrainNetworkAll()
print("Error for the sequential network = ",NetworkNorm.Distance())

NetworkNorm.PlotOutput("Sequential_Training")


#%% Testing - Amount of Nodes


ErrorsM = []
TimeList = []
Mlist = np.linspace(1,50,50,dtype=np.int)

for M in Mlist:
    print(M)
    NetworkTestM = NeuralNW(X,Y,M=M)
    now = time()
    for i in range(100):
        NetworkTestM.TrainNetworkRand()
    TimeList.append(time()-now)
    ErrorsM.append(NetworkTestM.Distance())

plt.figure("Erors M")
plt.plot(Mlist,ErrorsM)
plt.xlabel('M')
plt.ylabel('Error')

plt.figure("Time M")
plt.plot(Mlist,TimeList)
plt.xlabel('M')
plt.ylabel('t(s)')
#%% Testing - Eta 
Errorseta = []
EtaLog = np.linspace(-4,2,50)

Etas = np.power(10,EtaLog)

for eta in Etas:
    print(eta)
    NetworkTesteta = NeuralNW(X,Y,eta=eta)
    for i in range(100):
        NetworkTesteta.TrainNetworkRand()
    Errorseta.append(NetworkTesteta.Distance())

plt.figure("Errors eta")
plt.plot(Etas,Errorseta)
plt.xlabel('$\eta$')
plt.ylabel('Error')

#%% Error Weights
ErrorW = []
Weights = []
D = 2
M = 8
for i in range(-5,5):
    Weights.append([])
    for j in range(10):
        w1 = np.random.random_sample((D+1,M))-0.5+i
        w2 = np.random.random_sample(M+1)-0.5+i
        Weights[-1].append((w1,w2))


for i, partWeight in enumerate(Weights):
    print(i)
    ErrorW.append(0)
    for w1,w2 in partWeight:
        NetworkTestW = NeuralNW(X,Y)
        NetworkTestW.setWeights(w1,w2)
        for i in range(100):
            NetworkTestW.TrainNetworkRand()
        ErrorW[-1] += NetworkTestW.Distance()

ErrorW = np.array(ErrorW)/10

WPlot = range(-5,5)
plt.figure("Errors in W")
plt.plot(WPlot,ErrorW)
plt.xlabel('Inital weights bias')
plt.ylabel('Error')
        






#%%


Data = np.genfromtxt('data/a017_NNpdfGaussMix.txt')
X2 = Data[:,0:2]
Y2 = Data[:,2]
Plot(X2,Y2,"2.5")




#%%

RealDataNNW = NeuralNW(X2,Y2,eta=0.01,M=80)



for i in range(2000):
    
    if (i % 20 == 0):
        RealDataNNW.PlotOutput("Real" + str(i),save=True,savename="../latex/Images/Final2/" + str(i) + ".png",title="Iteration " + str(i))
    RealDataNNW.TrainNetworkRand()


RealDataNNW.PlotOutput("Real" + str(i),save=True,savename="../latex/Images/Final2/" + str(i+1) + ".png",title="Iteration " + str(i+1))


#%% This is to test the convergence for different Eta's in case it often overshoots --> Spoiler: that wasn't the case (Warning: Takes a long time to compute!)
