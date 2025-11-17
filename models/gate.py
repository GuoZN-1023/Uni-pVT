import torch
import torch.nn as nn
from .experts import get_activation

class GateNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act = get_activation(activation)
        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h), act]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.net(x))