# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:05:57 2019

@author: Balint
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import copy
import matplotlib.pyplot as plt

#%%
# Keras imports
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Input, Dropout, concatenate, add
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, l1_l2


#%%

valid_split = 0.2
test_split = 0.1


#%% 
## ADATBETOLTES
df = pd.read_hdf('processed_data.hdf5', key='imdb')

#%%
X = df.loc[:,df.columns != 'imdb_score'].values
Y = (df['imdb_score']).values
Y = Y.reshape(-1,1)
#%%
Y.shape

#%%
X_train, X_valid, X_test = X[0:int(len(X)*(1-valid_split-test_split))], \
                           X[int(len(X)*(1-valid_split-test_split)):int(len(X)*(1-test_split))], \
                           X[int(len(X)*(1-test_split)):]
Y_train, Y_valid, Y_test = Y[0:int(len(X)*(1-valid_split-test_split))], \
                           Y[int(len(X)*(1-valid_split-test_split)):int(len(X)*(1-test_split))], \
                           Y[int(len(X)*(1-test_split)):]
#%%%    
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

#%%%

print("SkĂĄlĂĄzĂĄs utĂĄn")
print("Train mean:",np.mean(X_train),", std:",np.std(X_train ))
print("Valid mean:",np.mean(X_valid),", std:",np.std(X_valid ))
print("Test mean:",np.mean(X_test),", std:",np.std(X_test))

#%%


es = EarlyStopping(patience=10)
mcp = ModelCheckpoint(filepath='model_imdb_residual.hdf5',save_best_only=True)

inputs = Input(shape=(X_train.shape[1],))

ndense = 2
nblocks = 5
nres = 2

x = inputs
for k in range(nblocks):
    res2 = Dense(16, activation='tanh')(x)
    for j in range(nres):        
        residual = Dense(16, activation='tanh')(x) #ez egy egyeduli reteg, az input es a blokk utolso layere kozott
        block = x
        for i in range(ndense):
            block = Dense(int(16/(i+1)), activation='tanh')(block)
            block = Dropout(0.5)(block)
        block = Dense(16, activation='tanh')(block)
        block = add([residual,block])
        x =  block
    block = add([res2,block])

y = Dense(1, activation='linear')(block)

model = Model(inputs, y)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit(X_train, Y_train,
          validation_data=(X_valid, Y_valid),
          epochs=1,
          callbacks=[es,mcp], verbose=1)

model = load_model('model_imdb_residual.hdf5')
        

#%%
print(model.summary())
#%%
#teszteles
err = model.evaluate(X_test,Y_test)
print("Teszt hiba:",err)

#%% 
# predikciĂł
preds = model.predict(X_test)
print(preds)
#%%

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
sns.regplot(Y_test, preds).set(xlim=(1,10),ylim=(1,10))

#%%
# HALO MEGJELENITESE
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))