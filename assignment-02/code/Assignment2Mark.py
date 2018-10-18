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

#%% Constants


Alpha = 90
Beta = 110

m = 0.6*1000 #People in favour
l = 0.4*1000 #People against

#%% Functions


def BetaDis(Mu,Alpha,Beta,exact=False):
    if(exact):
        Frac = np.math.gamma(Alpha + Beta)/(np.math.gamma(Alpha) * np.math.gamma(Beta))
    else:
        FracLog = LogFactorial(Alpha + Beta-1) - (LogFactorial(Alpha-1) + LogFactorial(Beta-1))
        Frac = np.exp(FracLog)
    return Mu**(Alpha-1)*(1-Mu)**(Beta-1)*Frac
    """
    The function can approximate the fraction involving Gamma if Alpha and Beta
    are whole, positive numbers. Gamma(x) = (x-1)!. When you calculate the Log
    of the factorial, the numbers won't get so high to create a 'math range error'.
    This is slightly more inaccurate but an error is typicall like 1 in 10^-9. 
       
    """

def LogFactorial(n):
    if (n is 1):
        return 0
    else:
        return np.log(n) + LogFactorial(n-1)

#%%

Mean = Alpha/(Alpha+Beta)
Variance = (Alpha*Beta)/((Alpha+Beta)**2*(Alpha+Beta+1))

print("The mean is\t" + '{:.2e}'.format(Mean))
print("The variance is\t" + '{:.2e}'.format(Variance))


#%% Plot of the distribution

x = np.linspace(0,1,10000)

y = BetaDis(x,Alpha,Beta)

plt.rcParams.update({'font.size': 22})

plt.xlabel("$\mu$",fontsize=50)

plt.plot(x,y)

#%%


