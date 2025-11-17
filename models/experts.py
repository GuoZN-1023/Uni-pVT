import torch
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class ExpertNetwork(nn.Module):
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
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)