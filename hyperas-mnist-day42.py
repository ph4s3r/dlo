# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:53:20 2019

@author: Balint
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l1, l2

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(28*28,)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(10, activation='softmax'))


#%%# adatok
from keras.datasets import mnist
from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#%%
X_train.shape
#%%
for i in range(10):
    plt.figure(figsize=(2,2))
    plt.imshow(X_train[i])
#%%
print(Y_train[:10])

#%%
import seaborn as sns
sns.countplot(Y_train)
#%%
sns.countplot(Y_test)
#%%
### adatok elokeszitese
## bemenetek 
X_train = X_train.astype('float32')
X_test=   X_test.astype('float32')

X_train = X_train / 255
X_test  = X_test / 255 
Y_train = np_utils.to_categorical(Y_train)
Y_test  = np_utils.to_categorical(Y_test)
#%%
#%%%
X_train = X_train.reshape(60000,-1)
X_test  = X_test.reshape(-1,28*28)

#%%
X_train[0]
#%%%


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), \
              metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split=0.2, epochs=25)


#%%

sns.distplot(model.layers[1].get_weights()[0].reshape(-1), kde=False)

#%%
np.max(model.layers[1].get_weights()[0])

#%%
weights_l1 = model.layers[1].get_weights()[0]

#%%
len(weights_l1.reshape(-1))
#%%
len(weights_l1[weights_l1>10e-4])


#%%

from keras import backend as K
def get_activation(model, layer, X_batch):
    activation_f = K.function([model.layers[0].input, K.learning_phase()],[layer.output,])
    activation = activation_f((X_batch, False))
    return activation

#%%
l1_features = get_activation(model, model.layers[0], X_test)
#%%

l1_features[0].shape

#%%

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

tsne_features = TSNE(n_components=2).fit_transform((l1_features[0])[:4000])
#%%
plt.figure(figsize=(10,10))
#plt.scatter(tsne_features[:,0], tsne_features[:,1], c=plt.cm.jet(np.argmax(Y_test[:4000],axis=1)/10), s=10, edgecolors='none')
plt.scatter(tsne_features[:,0], tsne_features[:,1])
plt.show()

#%%

l2_features = get_activation(model, model.layers[1], X_test)
#%%

l2_features[0].shape

#%%

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

tsne_features = TSNE(n_components=2).fit_transform((l2_features[0])[:4000])
#%%
plt.figure(figsize=(10,10))
plt.scatter(tsne_features[:,0], tsne_features[:,1], c=plt.cm.jet(np.argmax(Y_test[:4000],axis=1)/10), s=10, edgecolors='none')
plt.show()
#%%
model.summary()
#%%

#%%

preds = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(preds,axis=1))
plt.figure(figsize=(10,8))
sns.heatmap(conf, annot=True, fmt='.0f')

#%%
rossz5os = preds[(np.argmax(Y_test,axis=1)==5) & (np.argmax(preds,axis=1)==3)]
rossz5os_kep = X_test[(np.argmax(Y_test,axis=1)==5) & (np.argmax(preds,axis=1)==3)]
#%%
len(rossz5os)

#%%
print(rossz5os)
#%%
rossz5os[0]

#%%
for i in range(len(rossz5os)):
    plt.figure(figsize=(4,4))
    plt.imshow(rossz5os_kep[i].reshape(28,28))