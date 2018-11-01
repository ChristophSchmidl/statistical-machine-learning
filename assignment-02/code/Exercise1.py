# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:34:24 2018

@author: mbeijer
"""

import numpy 
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
PC0ZLow = (16/81)*np.log(1.3) - 1/30
PC1ZLow = (4/9)*np.log(1.3)-2/45

MisFirst = PC0ZLow/PC1ZLow

PC0ZHigh = (4/9)*( ((4/9)*np.log(4)-(1/3)) - ( (4/9) *np.log(1.3) - (1/30))  )
PC1ZHigh = (4/3)*(( (1/3)*np.log(4) - 1 ) - ( (1/3)* np.log(1.3)- 0.1)  )

MisSecond = PC1ZHigh/PC0ZHigh
