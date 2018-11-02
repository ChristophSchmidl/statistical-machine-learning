# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:34:24 2018

@author: mbeijer
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

def ZWithC0(z):
    return (3/2)*(1-z**2)
    
def ZWithC1(z):
    return (6/5)*z*(z+1)
    
def C0WithZ(z):
    return (4/9)*(1-z)/(3*z+1)

def C1WithZ(z):
    return (4*z)/(3*z+1)

def C0Int(zhigh,zlow):
    return ((16/81)*np.log(3*zhigh+1)-(4/27)*zhigh) - ((16/81)*np.log(3*zlow+1)-(4/27)*zlow)

def C1Int(zhigh,zlow):
    return ((4/3)*zhigh)-(4/9)*np.log(3*zhigh+1) - ((4/3)*zlow) - (4/9)*np.log(3*zlow+1)

def zInt(zhigh,zlow):
    return (1/4)*(zhigh**3+2*zhigh**2+zhigh) - (1/4)*(zlow**3+2*zlow**2+zlow)
#%%

z = np.linspace(0,1,100000)

y0 = ZWithC0(z)
y1 = ZWithC1(z)

plt.plot(z,y0,label="z with claim being false")
plt.plot(z,y1,label="z with claim being true")

plt.xlabel("z")
plt.ylabel("p(z|c)")


plt.legend()


#%%


y0 = C0WithZ(z)
y1 = C1WithZ(z)

plt.plot(z,y0,label="Change of c=0 for 0 < z < 1")
plt.plot(z,y1,label="Change of c=1 for 0 < z < 1")


plt.xlabel("z")
plt.ylabel("p(c|z)")


plt.legend(loc='middle right')

#%%

Diff = y0-y1
Diff = Diff**2

LocInter = np.where(Diff == np.min(Diff))

print("The intersection is at z = " + str(z[LocInter[0]]))

#%%%
PC0ZLow = C0Int(0.1,0)
PC1ZLow = C1Int(0.1,0)

MisFirst = PC1ZLow/PC0ZLow

PC0ZHigh = C0Int(1,0.1)
PC1ZHigh = C1Int(1,0.1)

MisSecond = PC0ZHigh/PC1ZHigh

print("Integral for C=0 from 0 to 0.1 is \t\t\t" + str(PC0ZLow))
print("Integral for C=1 from 0 to 0.1 is \t\t\t" + str(PC1ZLow))
print("The misclassification error for the first part is: \t" + str(MisFirst) + "\n")

print("Integral for C=0 from 0 to 0.1 is \t\t\t" + str(PC0ZHigh))
print("Integral for C=1 from 0 to 0.1 is \t\t\t" + str(PC1ZHigh))
print("The misclassification error for the first part is: \t" + str(MisSecond))

zlowInt = zInt(0.1,0)
zhighInt = zInt(1,0.1)

print("Integral of z for low z = \t" + str(zlowInt))
print("Integral of z for high z = \t" + str(zhighInt))

TotalError = MisFirst*zlowInt + MisSecond*zhighInt

print("The total misclassificaiton error is " + str(TotalError))

#%%

