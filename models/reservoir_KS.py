import torch
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import scipy


### PARAMETERS ###
T = 80000
r_spectral = 0.4
learning_rate = 1e-6
device = torch.device("cpu")
D_in, H, D_out = 65, 1000, 65
dtype = torch.double


def check():
    import ipdb
    ipdb.set_trace()

def set_spectral_radius(W,  r_spectral, device=torch.device('cpu')):
    # larger spectral-radius implies longer-range interactions   
    # spectral radius > 1 has no echo state property
    
    # to-numpy
    W = W.to(torch.device('cpu')).numpy()
    #W *= 100
    #W.astype(int)
    
    # current radius
    import time
    
    print(H)
    # numpy 
    tic = time.time()
    r_initial = max(abs(np.linalg.eigvals(W)))
    tac = time.time()
    print('numpy:', tic-tac, r_initial)

    # scipy
    tic = time.time()
    r_initial = abs(scipy.sparse.linalg.eigs(W, k=1, return_eigenvectors=False))
    tac = time.time()
    print('scipy:', tic-tac, r_initial)

    # torch cpu
    W = torch.from_numpy(W)
    tic = time.time()
    r_initial = torch.max(torch.abs(torch.eig(W, eigenvectors=False)[0]))
    toc = time.time()
    print('torch-cpu:', tic-toc, r_initial)

    # torch gpu
    W.to(torch.device('cuda'))
    tic = time.time()
    r_initial = torch.max(torch.abs(torch.eig(W, eigenvectors=False)[0]))
    tac = time.time()
    print('torch-gpu:', tic-tac, r_initial)

    # scale
    W_scaled = W * (r_spectral / r_initial)
    # to-Tensor
    #W_scaled = torch.from_numpy(W_scaled)
    
    return W_scaled.to(device)



### PREPROCESS DATA ###
def preprocess_data(infile='../data/kuramoto-sivashinsky/ut.npy'):
    global T, device, dtype
    print('### PREPROCESS DATA ###')

    # read
    ut = np.load(infile) # array, float64, (65, 100001)
    print('... load array ut:', ut.shape)

    # normalize
    ut = preprocessing.scale(ut) # std=1, mean=0
    print('... nomalize ut:', ut.shape, 'std=', round(ut.std()), 'mean=', round(ut.mean()))

    # transpose
    ut = ut.T # (100001, 65)

    # train test split
    train, test = ut[:T], ut[T:] # (80000, 65), (20001, 65)
    print('... train test split', train.shape[0], test.shape[0])

    # to-Tensor
    train, test = torch.from_numpy(train), torch.from_numpy(test)
    print('... to-tensor')
    print('\t train, test', train.shape, test.shape) # (80000, 65), (20001, 65)

    return (train.to(device), test.to(device)) 

print('save')
torch.save(preprocess_data(), '../data/kuramoto-sivashinsky/normalized_ut')
print('load')
train, test = torch.load('../data/kuramoto-sivashinsky/normalized_ut')
print('done')


### BUILD MODEL ####

print('... initialize model')
# weights
Wih = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=False) 
Whh = torch.randn(H, H, device=device, dtype=dtype, requires_grad=False)
Who = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# set spectral radius
print('... set spectral radius')
Whh = set_spectral_radius(Whh, r_spectral, device=device)
import sys
sys.exit()
### IMPLEMENT TRAINING LOOP ###
sweeps = 1
print('... starting train loop with {} sweeps'.format(sweeps))
for sweep in range(sweeps):
    print('... sweep: {}'.format(sweep))

    # clear memory
    prev_h = torch.zeros(1, H, device=device, dtype=dtype) # clear memory
    
    # train
    for t in range(train.shape[1] - 1):
        input, label = train[t], train[t+1]
        print(input.shape, Wih.shape)
        i2h = torch.mm(input.reshape(1,-1), Wih)
        h2h = torch.mm(prev_h, Whh)
        next_h = i2h + h2h
        next_h = next_h.tanh()
        prev_h = next_h
        output = torch.mm(prev_h, Who)
        output = output.tanh()

        loss = torch.sum((output - label)**2)

        loss.backward()
        with torch.no_grad():
            Who -= learning_rate * Who.grad
            Who.grad.zero_()
        if t%int(train.shape[0]/1000)==0:
            print(int(t/(train.shape[0]/1000)), loss)
        if t > 1000:
            break
    # test 
