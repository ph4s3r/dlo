# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:09:25 2019

@author: karapet
"""
import numpy as np

seged = np.linspace(-4*np.pi, 4*np.pi, 1000)
sinx = np.sin(seged)
print(sinx.shape)
sinx2 = (sinx+np.random.normal(0,1,1000))*2
import matplotlib.pyplot as plt
plt.plot(seged, sinx,'b',seged,sinx2,'r')