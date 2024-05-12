from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, input_length, input_dim, hidden_dim, output_dim):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.input_length = input_length
        self.x_h = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.h_o = nn.Linear(hidden_dim,output_dim)



    def forward(self, x):
        # Implementation here ...
        device = next(self.parameters()).device
        batch_size = x.size(0)
        h = torch.zeros(1,self.hidden_dim)
        h = h.expand(batch_size,-1).to(device)
        for i in range(self.input_length):
            x_i = x[:, i, :]
            input_combined = torch.cat((x_i, h), dim=1)
            h = torch.tanh(self.x_h(input_combined))
        out = self.h_o(h)
        return out
#     # add more methods here if needed
# import torch
# import torch.nn as nn

# class VanillaRNN(nn.Module):

#     def __init__(self, input_length, input_dim, hidden_dim, output_dim):
#         super(VanillaRNN, self).__init__()
#         self.input_length = input_length
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim

#         # Initialize parameters
#         self.W_hx = nn.Parameter(torch.randn(hidden_dim, input_dim))
#         self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         self.b_h = nn.Parameter(torch.zeros(hidden_dim))
#         self.W_ph = nn.Parameter(torch.randn(output_dim, hidden_dim))
#         self.b_o = nn.Parameter(torch.zeros(output_dim))

#     def forward(self, x):
#         # Initialize hidden state
#         device = next(self.parameters()).device
#         h = torch.zeros(1, self.hidden_dim).to(device)

#         # Iterate through time steps
#         for t in range(self.input_length):
#             # Update hidden state
#             h = torch.tanh(torch.matmul(self.W_hx, x[:, t, :].T) + torch.matmul(self.W_hh, h.T) + self.b_h.unsqueeze(1))
        
#         # Compute output
#         o = torch.matmul(self.W_ph, h.T) + self.b_o.unsqueeze(1)
#         y = torch.softmax(o, dim=0)

#         return y
