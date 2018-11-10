import torch
import numpy as np
from rc import RNN
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


### PARAMETERS ###

lenX = 1
lenY = 1
learning_rate = 1e-6


### PREPROCESS DATA ###
print('### PREPROCESS DATA ###')

# read
ut = np.load('ut.npy') # array, float64, (65, 100001)
print('... load array ut:', ut.shape)

# normalize
ut = preprocessing.scale(ut) # std=1, mean=0
print('... nomalize ut:', ut.shape, 'std=', round(ut.std()), 'mean=', round(ut.mean()))

# slice
xlist, ylist = [], []
for i in range(ut.shape[1] - lenX):
    X, Y = ut[:,i:lenX + i], ut[:,i+lenX:i+lenX+lenY] # (65, 16), (65, 1)
    xlist.append(X) 
    ylist.append(Y)

# to-numpy
xlist, ylist = np.stack(xlist), np.stack(ylist) # (99985,65,16), (99985,65,1)
print('... slice ut to:\t\t\t x.shape=', xlist.shape, 'y.shape=', xlist.shape)

# train test split
xtrain, xtest, ytrain, ytest = train_test_split(xlist, ylist, test_size=0.2, random_state=4) # (79988,65,16), (19997,65,16), (79988,65,1), (19997,65,1)
print('... xtrain, xtest, ytrain, ytest:\t', xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

# to-Tensor
xtrain, xtest, ytrain, ytest = torch.from_numpy(xtrain), torch.from_numpy(xtest), torch.from_numpy(ytrain), torch.from_numpy(ytest)
print('... to-tensor:\t\t\t\t', xtrain.shape, xtest.shape)



### BUILD MODEL ####

from rc import RNN
model = RNN(input_size=65*16, hidden_size=200, output_size=65*1)


### IMPLEMENT TRAINING LOOP ###

