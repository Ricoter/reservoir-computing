"""
    Reservoir computer by Rico van Midde 2018
    possible extra: i20 layer as in ./rnn.py
"""

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__() # avoid referring to the base class explicitly ?
        self.hidden_size = hidden_size

        #self.i2h = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        #self.h2h = nn.Linear(hidden_size, hidden_sizedd, bias=False)
        self.o2o = nn.Linear(hidden_size, output_size, bias=False)        
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def set_spectral_radius(self, r_spectral):
        if r_spectral>1:
            print("Warning: spectral radius > 1 has no echo state property!")
        
        # to-numpy
        Wh = torch.numpy(self.h2h.weights)
        # current radius
        r_initial = max(abs(np.linalg.eigvals(Wh)))
        # scale
        Wh_scaled = Wh * (r_spectral / scale_factor)
        # to-Tensor
        Wh_scaled = torch.from_numpy(Wh_scaled)

        self.h2h.weights = torch.nn.Parameter(initial_param['weight'])


    def print_

    def initHidden(self):
        return torch.zeros(1, self.hidden_size) # (1, hidden_size)
