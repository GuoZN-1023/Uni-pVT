# models/gate.py
import torch
import torch.nn as nn
from .experts import get_activation


class GateNetwork(nn.Module):
    """
    门控网络：
      - 输入：与专家相同的特征向量 x
      - 输出：对 n_experts 个专家的 softmax 权重
    """
    def __init__(self, input_dim, hidden_layers,
                 activation: str = "relu",
                 dropout: float = 0.0,
                 n_experts: int = 4):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act = get_activation(activation)
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_experts))
        self.net = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.net(x)
        w = self.softmax(logits)  # [B, n_experts]
        return w
