import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import copy
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, l1_l2
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def data22():
    valid_split = 0.2
    test_split = 0.1
    df = pd.read_hdf('processed_data.hdf5', key='imdb')
    X = df.loc[:,df.columns != 'imdb_score'].values
    Y = (df['imdb_score']).values
    Y = Y.reshape(-1,1)
    X_train, X_valid, X_test = X[0:int(len(X)*(1-valid_split-test_split))], \
                               X[int(len(X)*(1-valid_split-test_split)):int(len(X)*(1-test_split))], \
                               X[int(len(X)*(1-test_split)):]
    Y_train, Y_valid, Y_test = Y[0:int(len(X)*(1-valid_split-test_split))], \
                               Y[int(len(X)*(1-valid_split-test_split)):int(len(X)*(1-test_split))], \
                               Y[int(len(X)*(1-test_split)):]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)
    print("Train mean:",np.mean(X_train),", std:",np.std(X_train ))
    print("Valid mean:",np.mean(X_valid),", std:",np.std(X_valid ))
    print("Test mean:",np.mean(X_test),", std:",np.std(X_test))
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
    
def model123(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    es = EarlyStopping(patience=10, min_delta=10e-3)
    mcp = ModelCheckpoint(filepath='model_imdb_hyperas.hdf5',save_best_only=True)
    
    model = Sequential()
    model.add(Dense({{choice([25,50,75,100,200])}}, input_dim=X_train.shape[1]))
    model.add(Activation({{choice(['sigmoid','tanh','relu'])}}))
    model.add(Dropout({{choice([0,0.25,0.5])}}))
    if {{choice(['one','two'])}} == 'two':
        model.add(Dense({{choice([25,50,75,100,200])}}, kernel_regularizer={{choice([l1(0.01),l2(0.01),l1(0.1),l2(0.1)])}}))
        model.add(Activation({{choice(['sigmoid','tanh','relu'])}}))
    model.add(Dropout({{choice([0,0.25,0.5])}}))
    model.add(Dense(Y_train.shape[1]))
    
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr={{uniform(0.001,0.05)}}))
    
    model.fit(X_train, Y_train,
              validation_data=(X_valid, Y_valid),
              epochs=100000000000,
              callbacks=[es,mcp], verbose=0)

    test_err = model.evaluate(X_test, Y_test, verbose=0)
    print("Test error:",test_err)
    return {'loss': test_err, 'status': STATUS_OK, 'model': model}
    

    
best_run, best_model = optim.minimize(model=model123, data=data22,
                                      algo=tpe.suggest, max_evals=100,
                                      trials=Trials())

#%%
best_model.summary()

#%%
print(best_run)

#%%
for layer in best_model.layers:
    print("RĂŠteg:", layer.name)
    print(layer.get_config())
    print('\n')