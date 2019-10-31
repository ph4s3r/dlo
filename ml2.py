# neuralnet = MLP(2,10,10,1)
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
#%%
def activation(x):
    return 1 / (1 + np.exp(-x))

def dactivation(x):
    return np.exp(-x)/((1+np.exp(-x))**2)
#%%
segedx = np.linspace(-6,6,200)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(segedx, activation(segedx),'b',segedx, dactivation(segedx),'r') 

#%%
segedx
#%%
class MLP:
    def __init__(self, *args):
        np.random.seed(123)
        self.shape = args # (2,3,1)
        n = len(args)
        self.layers = []
        # bemeneti reteg
        # BIAS --- TODO: ebedszunet utan, miert kell a bias??!! - szztem hogy ne legyen nullaval szorzas..
        self.layers.append(np.ones(self.shape[0]+1))
        for i in range(1,n):
            self.layers.append(np.zeros((self.shape[i])))
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))
        #self.dw = [0,]*len(...)
        self.reset()
    
    def reset(self):
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,
                                  self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*1

    def propagate_forward(self, data):
        self.layers[0][0:-1] = data
        for i in range(1,len(self.shape)):
            self.layers[i][...] = activation(np.dot(self.layers[i-1],
                                                     self.weights[i-1]))
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []
        error = -(target-self.layers[-1]) # y - y_kalap
        delta = np.multiply(error, dactivation(np.dot(self.layers[-2],
                                                     self.weights[-1])))
        deltas.append(delta)
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)* \
                    dactivation(np.dot(self.layers[i-1],self.weights[i-1]))
            deltas.insert(0,delta)
            
        # sĂşlyok mĂłdosĂ­tĂĄsa
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = -lrate*np.dot(layer.T,delta)
            self.weights[i] += dw 
            #self.dw[i] = dw

        return (error**2).sum()
		
		
####
### XOR adatok letrehozasa
# XOR: 
# x1   x2    y
# 0    0     0
# 0    1     1
# 1    0     1
# 1    1     0
#%%
samples = np.zeros(200, dtype=[('input', float, 2), ('output',float,1)])
for i in range(0,200,4):
    noise = np.random.normal(0,1,8) # 0 v.e., 1 szoras, 8 adatpont
    samples[i]   = (-2+noise[0],-2+noise[1]), 0
    samples[i+1] = (2+noise[2],-2+noise[3]), 1
    samples[i+2] = (-2+noise[4],2+noise[5]), 1
    samples[i+3] = (2+noise[6],2+noise[7]), 0

plt.figure()
plt.scatter(samples['input'][:,0],samples['input'][:,1],c=samples['output'][:])

best_neuralnet = MLP(2,3,1)
####

def learn(network,samples, valid_split=0.2, test_split=0.1, epochs=10000, lrate=0.01, momentum=0.9, patience=10):

        samples_train = samples[0:int(len(samples)*(1-valid_split-test_split))]
        samples_valid = samples[int(len(samples)*(1-valid_split-test_split)):int(len(samples)*(1-test_split))]
        samples_test  = samples[int(len(samples)*(1-test_split)):]
    
        scaler = StandardScaler().fit(samples_train['input'])
        samples_train['input'] = scaler.transform(samples_train['input'])
        samples_valid['input'] = scaler.transform(samples_valid['input'])
        samples_test['input'] = scaler.transform(samples_test['input'])
    
        np.random.shuffle(samples_train)
        
        min_MSE = 1
        patience_counter = 0
    
        for i in range(epochs):
            ## TRAIN ADATOK
            train_err = 0
            for k in range(samples_train.size):
                network.propagate_forward( samples_train['input'][k] )
                train_err += network.propagate_backward( samples_train['output'][k], lrate, momentum )
            train_err /= samples_train.size

            #VALIDACIO
            valid_err = 0
            o_valid = np.zeros(samples_valid.size)
            for k in range(samples_valid.size):
                o_valid[k] = network.propagate_forward( samples_valid['input'][k])
                valid_err += (o_valid[k]-samples_valid['output'][k])**2
            valid_err /= samples_valid.size

            print("%d epoch, train_err: %.4f, valid_err: %.4f" % (i, train_err, valid_err))
            rve = round(valid_err,6)
            if(min_MSE>rve):
                min_MSE = rve
            else:
                patience_counter+=1
            
            if(patience_counter>patience):
                best_neuralnet = neuralnet
                print("Early stop")
                break

        print("min MSE:", min_MSE)    
        print("patience_counter", patience_counter)
        test_err = 0
        o_test = np.zeros(samples_test.size)
        for k in range(samples_test.size):
            o_test[k] = network.propagate_forward( samples_test['input'][k])
            test_err += (o_test[k]-samples_test['output'][k])
            print(k, samples_test['input'][k], '%.2f' % o_test[k], \
                  ' (elvart eredmeny: %.2f)' % samples_test['output'][k])
        test_err /= samples_test.size
        
        print("Test MSE:",test_err)
    
        fig1=plt.figure()
        plt.scatter(samples_test['input'][:,0], samples_test['input'][:,1], c=np.round(o_test[:]), cmap=plt.cm.cool)

neuralnet = MLP(2,3,1)
learn(neuralnet,samples)