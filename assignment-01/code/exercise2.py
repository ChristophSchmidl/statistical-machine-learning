# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:11:47 2018

@author: Mark
"""

#%%Exercise 2


#%%Imorts
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
import random as rand
from mpl_toolkits.mplot3d import Axes3D


#%%Functions

def H(x,y):
    return 100*(y-x**2)**2+(1-x)**2

def NabH(x,y): #Gives the nabla for a given point (x,y).
    return np.array([400*x**3-400*x*y+2*x-2,200*y-200*x**2])

def NabHvec(X): 
    return NabH(X[0],X[1])
    
def Distance(X,Y): #Calculates the distance between two (two-dimensional) vectors.
    return np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2)

def TooFar(X): #If the distance from the origin is larger then 10^10 this functions returns true. This because if we wander that far from the origin the next steps will lead us to values to high to calculate.
    return Distance(X,[0,0]) > 10**10

def Test(eta,StepTimes,TestTimes,xmin,xmax,ymin,ymax,MinDistance,Best=[1,1]):
    Result = 0
    for i in range(TestTimes):
        RandBegin = np.array([rand.uniform(xmin,xmax),rand.uniform(ymin,ymax)])
        for j in range(StepTimes):
            NewPoint = RandBegin - eta * NabHvec(RandBegin)
            RandBegin = NewPoint
            if(TooFar(RandBegin)):
                break
        if(Distance(RandBegin,Best) < MinDistance):
            Result += 1
    return Result
    
def save2D(X,FileName): #Saves a 2d array X to a file with FileName name.
    File = open(FileName,'w')
    for i in range(len(X)):
        for j in range(len(X[0])):
            File.write(str(X[i][j]))
            if(j < (len(X[0])-1)):
                File.write('\t')
        if(i < (len(X)-1)):
            File.write('\n')
    File.close()

def save1D(X,FileName):#Saves a 1d array X to a file with FileName name.
    File = open(FileName,'w')
    for i in range(len(X)):
        File.write(str(X[i]))
        if(i < (len(X) -1)):
            File.write('\n')
    File.close()
#%%Constants
    
ReCalculate = False

Amount = 1000
xmin = -2
xmax = 2
ymin = -1
ymax = 3

Eta = 2*10**-3
Punt = np.array([-1,2])

LogEtasMin = -7
LogEtasMax = -2
AmountEtas = 100

StepTimesMin = 1
StepTimesMax = 300
MinDistance = 0.1
TestTimes = 100



#%% Create Data
   

x = np.linspace(xmin,xmax,Amount)
y = np.linspace(ymin,ymax,Amount)
x,y = np.meshgrid(x,y)
z = H(x,y)

#%%TEST


Best = np.array([1,1])

fig = plt.figure("Path over 3d surface")
ax = fig.gca(projection='3d')

plt.xlabel('x',fontsize=20,labelpad=20)
plt.ylabel('y',fontsize=20,labelpad=20)





surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)



fig.colorbar(surf, shrink=0.5, aspect=5)


ax.plot([1,1],[1,1],[0,2000],color='red')


for i in range(100):
    PuntNew = Punt - Eta*NabHvec(Punt)
    X = np.linspace(Punt[0],PuntNew[0],Amount)
    Y = np.linspace(Punt[1],PuntNew[1],Amount)
    Z = H(X,Y)
    ax.plot(X, Y, Z, color='orange')
    Punt = PuntNew
plt.title('Path over 3d surface',fontsize=40)
plt.tick_params(labelsize=20)
matplotlib.rcParams.update({'font.size': 22}) 

#%% Num Test
"""
etas = np.linspace(1*10**-6,1*10**-2,100)
point = []
testMax = 1000


for eta in etas:
    point.append(0)
    for i in range(testMax):
        RandBegin = np.array([rand.uniform(xmin,xmax),rand.uniform(ymin,ymax)])
        for j in range(100):
            NewPoint = RandBegin - eta * NabHvec(RandBegin)
            RandBegin = NewPoint
            if(TooFar(RandBegin)):
                break
        if(Distance(RandBegin,Best) < 1):
            point[-1] += 1

plt.plot(etas,point)
"""
#%% Num Test 3d



if(ReCalculate):
    LogEtas = np.linspace(LogEtasMin,LogEtasMax,AmountEtas)
    Etas = 10**LogEtas
    StepTimes = np.linspace(StepTimesMin,StepTimesMax,StepTimesMax,dtype=int)
    
    Etas,StepTimes = np.meshgrid(Etas,StepTimes)
    Result = np.zeros_like(Etas)
    
    for i in range(len(Etas)):
        print(i)
        for j in range(len(Etas[0])):
            Result[i][j] = Test(Etas[i][j],StepTimes[i][j],TestTimes,xmin,xmax,ymin,ymax,MinDistance)
    save1D(LogEtas,'LogEtas.txt')
    save2D(Etas,'Etas.txt')
    save2D(StepTimes,'StepTimes.txt')
    save2D(Result,'Result.txt')
else:
    LogEtas = np.genfromtxt('LogEtas.txt')
    Etas = np.genfromtxt('Etas.txt')
    StepTimes = np.genfromtxt('StepTimes.txt')
    Result = np.genfromtxt('Result.txt')

fig2 = plt.figure("Eta vs Times vs Good")
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(LogEtas, StepTimes, Result, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

plt.title("Amount of sucessful walks for a given $\eta$ and amount of steps.")
plt.xlabel('LogEta',labelpad=20)
plt.ylabel('Time',labelpad=20)
fig2.colorbar(surf2, shrink=0.5, aspect=5)





#%%
    
"""



Nab = NabHvec(RandBegin)
xDif = Nab[0]
if(xDif < 0):
    Spacex = RandBegin[0] - xmax
else:
    Spacex = RandBegin[0] - xmin
etax = Spacex/xDif
yDif = Nab[1]
if(yDif < 0):
    Spacey = RandBegin[1] - ymax
else:
    Spacey = RandBegin[1] - ymin
etay = Spacey/yDif
eta = min(etay,etax)

"""