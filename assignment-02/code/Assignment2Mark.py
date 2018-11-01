# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:23:45 2018

@author: mbeijer
"""

"""
Assignment 2 - Exercise 2

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#%% Constants

#Exercise 2
Alpha = 90
Beta = 110

m = 0.6*1000 #People in favour
l = 0.4*1000 #People against

#Exercise 3

Sigma = np.matrix([[0.14,-0.3,0,0.2],[-0.3,1.16,0.2,-0.8],[0,0.2,1,1],[0.2,-0.8,1,2]])

Lambda = Sigma**-1
#%% Functions

TempTestList = []
def BetaDis(Mu,Alpha,Beta,exact=False):
    print("Alpha = " + str(Alpha) + "Beta = " + str(Beta))
    if(exact):
        Frac = np.math.gamma(Alpha + Beta)/(np.math.gamma(Alpha) * np.math.gamma(Beta))
        return Mu**(Alpha-1)*(1-Mu)**(Beta-1)*Frac
    else:
        FracLog = LogGamma(Alpha + Beta) - (LogGamma(Alpha) + LogGamma(Beta))
        FracAns = np.log(Mu)*(Alpha-1) + np.log(1-Mu)*(Beta-1) + FracLog
        return np.exp(FracAns)
    """
    The function can approximate the fraction involving Gamma if Alpha and Beta
    are whole, positive numbers. Gamma(x) = (x-1)!. When you calculate the Log
    of the factorial, the numbers won't get so high to create a 'math range error'.
    This is slightly more inaccurate but an error is typicall like 1 in 10^-9. 
       
    """
def BetaDisLog(Mu,Alpha,Beta):
    FracLog = LogGamma(Alpha + Beta) - (LogGamma(Alpha) + LogGamma(Beta))
    print(FracLog)
    return np.log(Mu)*(Alpha-1) + np.log(1-Mu)*(Beta-1) + FracLog


def LogGamma(n):

    Total = 0    
    while(n > 3):
        Total += np.log(n-1)
        n -= 1
    return Total + np.log(np.math.gamma(n))

#%%

Mean = Alpha/(Alpha+Beta)
Variance = (Alpha*Beta)/((Alpha+Beta)**2*(Alpha+Beta+1))

print("The mean is\t" + '{:.2e}'.format(Mean))
print("The variance is\t" + '{:.2e}'.format(Variance))


#%% Plot of the distribution


plt.figure("First plot")
x = np.linspace(0,1,10000)

y = BetaDis(x,Alpha,Beta)

plt.rcParams.update({'font.size': 22})

plt.xlabel("$\mu$",fontsize=50)

plt.plot(x,y)

#%% Plot of the first and second distribution


Mean2 = (Alpha + m)/(Alpha + Beta + l + m)
Variance2 = ( (Alpha + m)*(Beta + l) )/( (Alpha + Beta + l + m)**2*(Alpha + Beta + l + m + 1)  )
print("The new mean is\t" + '{:.2e}'.format(Mean2))
print("The new variance is\t" + '{:.2e}'.format(Variance2))

plt.figure("Second plot")

y2 = BetaDis(x,Alpha + m, Beta + l)

plt.xlabel("$\mu$",fontsize=50)


plt.plot(x,y,label="prior density")
plt.plot(x,y2,label="posterior density")

plt.legend()

#%%

AlphaMin = BetaMin = 0
AlphaMax = BetaMax = 1000
Amount = 10000


Alpha = np.linspace(AlphaMin,AlphaMax,Amount)
Beta = np.linspace(BetaMin,BetaMax,Amount)
Alpha,Beta = np.meshgrid(Alpha,Beta)


Mean3D = (Alpha+l)/(Alpha + Beta + l + m)
Var3D = ( (Alpha + m)*(Beta + l) )/( (Alpha + Beta + l + m)**2*(Alpha + Beta + l + m + 1)  )

fig = plt.figure("First 3d Plot")

ax = fig.gca(projection='3d')

surf = ax.plot_surface(Alpha, Beta, Mean3D, cmap=cm.coolwarm,
                       linewidth=1, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel(r"$\alpha$",fontsize=50)
plt.ylabel(r"$\beta$",fontsize=50)
ax.view_init(elev=80, azim=270)
fig2 = plt.figure("Second 3d Plot")
ax2 = fig2.gca(projection='3d')
surf2 = ax2.plot_surface(Alpha, Beta, Var3D, cmap=cm.coolwarm,
                       linewidth=1, antialiased=True)
fig2.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel(r"$\alpha$",fontsize=50)
plt.ylabel(r"$\beta$",fontsize=50)
ax2.view_init(elev=80, azim=270)




