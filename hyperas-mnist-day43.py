# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:53:20 2019

@author: Balint
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

#%%# adatok
#from keras.datasets import mnist
from keras.utils import np_utils

#%%

    
#%%
def data42():
    f = open('mnist.npz', 'rb')
    data = np.load(f, encoding='bytes', allow_pickle=True)
    (X_train, Y_train), (X_test, Y_test) = (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])
    f.close()
    X_train = X_train.astype('float32')
    X_test=   X_test.astype('float32')
    X_train = X_train / 255
    X_test  = X_test / 255 
    Y_train = np_utils.to_categorical(Y_train)
    Y_test  = np_utils.to_categorical(Y_test)
    X_train = X_train.reshape(60000,-1)
    X_test  = X_test.reshape(-1,28*28)  
    
    return X_train, Y_train, X_test, Y_test

#%%
def model42(X_train, Y_train, X_test, Y_test):
    es = EarlyStopping(patience=10, min_delta=10e-3)
    mcp = ModelCheckpoint(filepath='model_mnist_hyperas.hdf5',save_best_only=True)
    
    model = Sequential()
    model.add(Dense({{choice([128,256,384])}}, activation='relu', input_shape=(28*28,)))
    model.add(Dropout({{choice([0.25,0.5])}}))
    model.add(Dense({{choice([128,256,384])}}, activation='relu', input_shape=(28*28,)))
    model.add(Dropout({{choice([0.25,0.5])}}))
    if {{choice(['two','three'])}} == 'three':
        model.add(Dense({{choice([128,256,384])}}, activation='relu', input_shape=(28*28,)))
        model.add(Dropout({{choice([0.25,0.5])}}))

    model.add(Dense(10, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), \
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              epochs=100,
              callbacks=[es,mcp],
              verbose=0)
    
    [loss, acc] = model.evaluate(X_test, Y_test, verbose=0)
    print("loss = ", loss)
    print("accuracy = ", acc)
    return {'loss': -loss, 'status': STATUS_OK, 'model': model}

#%%
#minimize-nal a minusz loss kell
best_run, best_model = optim.minimize(model=model42, data=data42,
                                      algo=tpe.suggest, max_evals=10,
                                      trials=Trials())

#%%
best_model.summary()

#%%
print(best_run)


_, _, x_test, y_test = data42()

best_model.evaluate(x_test, y_test, verbose=2)