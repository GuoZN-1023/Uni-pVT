# models/gate.py
import torch
import torch.nn as nn
from .experts import get_activation, ResidualBlock


class GateNetwork(nn.Module):
    """
    门控网络：
      - 输入：与专家相同的特征向量 x
      - 输出：对 n_experts 个专家的 softmax 权重

    结构：
      [ResidualBlock × N hidden] -> Linear -> Softmax
    """
    def __init__(
        self,
        input_dim: int,
        hidden_layers,
        activation: str = "relu",
        dropout: float = 0.0,
        n_experts: int = 4,
    ):
        super().__init__()

        blocks = []
        prev_dim = input_dim
        for h in hidden_layers:
            blocks.append(
                ResidualBlock(
                    in_dim=prev_dim,
                    out_dim=h,
                    activation=activation,
                    dropout=dropout,
                    use_batchnorm=True,
                )
            )
            prev_dim = h

        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.out = nn.Linear(prev_dim, n_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(x)
        logits = self.out(z)
        w = self.softmax(logits)  # [B, n_experts]
        return w