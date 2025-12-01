import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    """ Linear layer followed by the sine activation

    If `is_first == True`, then it represents the first layer of the network.
    In this case, omega_0 is a frequency factor, which simply multiplies the activations before the nonlinearity.
    Note that it influences the initialization scheme.

    If `is_first == False`, then the weights will be divided by omega_0 so as to keep the magnitude of activations constant,
    but boost gradients to the weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        # Initialize a linear layer with specified input and output features
        # 'bias' indicates whether to include a bias term
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    # initialize weights uniformly
    def init_weights(self):
        # diasble gradient calculation in initialization
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # Task 1 TODO
        # 1. pass input through linear layer (self.linear layer performs the linear transformation on the input)
        x = self.linear(input)

        # 2. scale the output of the linear transformation by the frequency factor
        x = x * self.omega_0

        # 3. apply sine activation
        x = torch.sin(x)

        return x


class Siren(nn.Module):
    """ SIREN architecture """

    def __init__(self, in_features, out_features, hidden_features=256, hidden_layers=3, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        # add the first layer
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        # append hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))


        if outermost_linear:
            # add a final Linear layer
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad(): # weights intialization
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            # otherwise, add a SineLayer
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net) # sequential wrapper of SineLayer and Linear

    def forward(self, coords):
        # coords represents the 2D pixel coordinates
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords