import torch
import numpy as np
#from rc import RNN
from torch.autograd import Variable
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def set_spectral_radius(W,  r_spectral, device=torch.device('cpu')):
    # larger spectral-radius implies longer-range interactions   
    # spectral radius > 1 has no echo state property

    # to-numpy
    W = W.to(torch.device('cpu')).numpy()
    # current radius
    r_initial = max(abs(np.linalg.eigvals(W)))
    # scale
    W_scaled = W * (r_spectral / r_initial)
    # to-Tensor
    W_scaled = torch.from_numpy(W_scaled)
    
    return W_scaled.to(device)


### PARAMETERS ###

lenX = 80000
lenY = 1
learning_rate = 1e-6
rnn = False
device = torch.device("cuda")
D_in, H, D_out = 65, 9000, 65
dtype = torch.double

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
print('... train_size, test_size:\t', xtrain.shape[0], xtest.shape[0])

# to-Tensor
xtrain, xtest, ytrain, ytest = torch.from_numpy(xtrain), torch.from_numpy(xtest), torch.from_numpy(ytrain), torch.from_numpy(ytest)
xtrain, xtest, ytrain, ytest = xtrain.permute(0,2,1), xtest.permute(0,2,1), ytrain.permute(0,2,1), ytest.permute(0,2,1) 
print('... to-tensor:\t xtrain, xtest, ytrain, ytest', xtrain.shape, xtest.shape, ytrain.shape, ytest.shape) # (79988, 16, 65), (19997, 16, 65), (79988, 1, 65), (19997, 1, 65)
xtrain.to(device), ytrain.to(device), xtest.to(device), ytest.to(device)


### BUILD MODEL ####

# weights
Wih = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=False) 
Whh = torch.randn(H, H, device=device, dtype=dtype, requires_grad=rnn)
Who = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# initialize hidden state
prev_h = torch.zeros(1, H, device=device, dtype=dtype)

# set spectral radius
Whh = set_spectral_radius(Whh, 0.4, device=device)

### IMPLEMENT TRAINING LOOP ###
for sweep in range(100):
    for i in range(xtrain.shape[0]):
        input = xtrain[i,:,:].to(device)
        label = ytrain[i,:,:].to(device)
        for t in range(xtrain.shape[1]):
            i2h = torch.mm(input[t,:].reshape(1,-1), Wih)
            h2h = torch.mm(prev_h, Whh)
            next_h = i2h + h2h
            next_h = next_h.tanh()
            prev_h = next_h
        output = torch.mm(prev_h, Who)
        output = output.tanh()
        loss = torch.sum((output - label)**2)

        loss.backward()
        with torch.no_grad():
            if rnn:
                Whh -= learning_rate * Whh.grad
            Who -= learning_rate * Who.grad
            Who.grad.zero_()
        if i%int(xtrain.shape[0]/1000)==0:
            print(int(i/(xtrain.shape[0]/1000)), loss)
