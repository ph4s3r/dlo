# -*- randcoding: utf-8 -*-
"""
Created on Mon Oct  7 15:04:46 2019

@author: karapet
"""
import numpy as np
import pandas as pd # DASK, RAPIDS.AI, BLAZING SQL

rng = pd.date_range("06/10/2019", periods=48, freq="30min")
print rng
df = pd.DataFrame(np.linspace(0,48*np.pi/12,48), index=rng, columns=["linspace"])
print df
df["masodik"] = np.sin(df['linspace'])
df["harmadik"] = df["masodik"]>=0
df.tail()


np.random.seed(123)
randindex = np.random.randint(0,len(df)-10)
print(randindex)

df.iloc[randindex:randindex+10]  = None


df['ip'] = df['masodik'].interpolate()

df['ip'].plot()