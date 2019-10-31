# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:42:30 2019

@author: karapet
"""

import numpy as np

def activation(x):
    return 1 / (1 + np.exp(-x))

def dactivation(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

#%%
segedx = np.linspace(-6,6,200)
import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(segedx, activation(segedx),'b',segedx, dactivation(segedx),'r') 
#%%

class MLP:
    def __init__(self, *args):
        np.random.seed(123)
        self.shape = args # (2,3,1) -> ez egy tuple
        n = len(args)
        self.layers = []
        # bemeneti reteg
        # BIAS --- TODO: ebedszunet utan, miert kell a bias??!! - szztem hogy ne legyen nullaval szorzas..
        self.layers.append(np.ones(self.shape[0]+1)) # listahoz adunk egy elemet, ami egy egyeseket tartalmazo akkora tomb, 
                                                        #amekkora a shape elso eleme, ami egy kettes (2,3,1)
                                                        #ez lesz szerintem az elso reteg (bemeneti reteg) hozzaadasa egyesek formajaban
        
        self.layers.append(np.ones(self.shape[0]+1))
        for i in range(1,n):
            self.layers.append(np.zeros((self.shape[i])))
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
        self.reset()
        
    def reset(self):
            for i in range(len(self.weights)):
                Z = np.random.random((self.layers[i].size,
                                  self.layers[i+1].size))
                self.weights[i][...] = (2*Z-1)*1
        
    def writeoutlayers(self):
            print("mlp1.shape", mlp1.shape)
            print "layers (type len and contents) \r\n\r\n", type(mlp1.layers), len(mlp1.layers), "\r\n\r\n", mlp1.layers, "\r\n\r\n" # just a list
            print "layers 1st element (type and contents) \r\n\r\n", type(mlp1.layers[0]), "\r\n\r\n", mlp1.layers[0], "\r\n\r\n" # just a list
            print "layers 2nd element (type and contents) \r\n\r\n", type(mlp1.layers[1]), "\r\n\r\n", mlp1.layers[1], "\r\n\r\n" # just a list
            print "layers 3rd element (type and contents) \r\n\r\n", type(mlp1.layers[2]), "\r\n\r\n", mlp1.layers[2], "\r\n\r\n" # just a list
            
    def writeoutweights(self):
            print "weights 1st element (type and contents) \r\n\r\n", type(mlp1.weights[0]), "\r\n\r\n", mlp1.weights[0], "\r\n\r\n" # just a list
            print "weights (type len and contents) \r\n\r\n", type(mlp1.weights), len(mlp1.weights), "\r\n\r\n", mlp1.weights, "\r\n\r\n" # just a list
            

# [] is a list
# () is a tuple
# {} is a dict
# self.shape is (2, 5, 9)
# self.shape[0] is 2)       
# self.layers is a list of arrays - 4 elements, and each is an array
# array([1., 1., 1.]), array([1., 1., 1.]), array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0.])] 
mlp1 = MLP(2,3,1)
mlp1.writeoutlayers()
mlp1.writeoutweights()


