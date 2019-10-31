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
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD
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

Y = np.round(Y)
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
X.shape
#%%
es = EarlyStopping(patience=10, min_delta=10e-3)
mcp = ModelCheckpoint(filepath='model_imdb.hdf5',save_best_only=True)

model = Sequential()
model.add(Dense(100, input_dim=X_train.shape[1]))
model.add(Activation('sigmoid'))
model.add(Dense(100, activation='sigmoid', kernel_regularizer=l2(0.01)))
#model.add(Dense(1000, activation='sigmoid'))
#model.add(Dense(1000, activation='sigmoid'))
#model.add(Dense(1000, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(Y_train.shape[1]), activation='softmax')
#model.add(Activation('sigmoid'))


#%%
print(model.summary())
#%%
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.01))

model.fit(X_train, Y_train,
          validation_data=(X_valid, Y_valid),
          epochs=100000000000,
          callbacks=[es,mcp])

#%%
import seaborn as sns
sns.distplot((model.layers[0].get_weights()[0]).reshape(-1), bins=20)
#%%
sns.distplot((model.layers[2].get_weights()[0]).reshape(-1), bins=20, kde=False)
#%%
model = load_model('model_imdb.hdf5')

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
