
from torch import nn
import numpy as np
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Simple function to init layers
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def make_nn_stack(
    input_size,
    output_size,
    hidden_sizes,
    nonlinearity,
    layer_norm=False,
    use_skip=False,
    fixup=False,
):
    """Create a stack of fully connected layers with nonlinearity"""
    sizes = [input_size] + hidden_sizes + [output_size]
    layers = []
    for i in range(1, len(sizes)):
        if fixup:
            layers.append(FixupLinear(sizes[i - 1], sizes[i]))
        else:
            layers.append(nn.Linear(sizes[i - 1], sizes[i]))

        if i < len(sizes) - 1:
            layers.append(nonlinearity)

        if layer_norm and i < len(sizes) - 1:
            layers.append(nn.LayerNorm(sizes[i]))

    if use_skip:
        return SkipConnectionStack(layers)
    else:
        return nn.Sequential(*layers)

class FixupLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FixupLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bias_a = nn.Parameter(torch.zeros(1))
        self.bias_b = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.fixup_init()

    def fixup_init(self):
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x + self.bias_a)
        x = x * self.scale + self.bias_b
        return x

class SkipConnectionStack(nn.Module):
    def __init__(self, layers):
        super(SkipConnectionStack, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if i > 0:
                    x = x + skip_connections[-1]
                skip_connections.append(x)
            x = layer(x)
        return x
