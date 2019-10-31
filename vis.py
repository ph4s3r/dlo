# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:00:30 2019

@author: karapet
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(123)

x = np.random.random(200)

print(x.shape)

sns.distplot(x)

x2 = np.random.normal(0,1,200)
#%%
sns.distplot(x2, kde=False) # kernel density


#%%
sns.distplot(x2, hist=False) # no histogram

mean, cov  =[0,1], [(1,.5),(0,5,1)]