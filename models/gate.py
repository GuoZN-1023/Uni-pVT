# models/gate.py
import torch
import torch.nn as nn
from .experts import ResidualBlock


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

        self.activation_name = activation
        self.n_experts = n_experts

        blocks = []
        prev_dim = input_dim
        for h in (hidden_layers or []):
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

        # 关键：gate 最后一层初始化为 0 -> softmax 初始接近均匀，避免早期“专家塌缩/偏置”
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(x)
        logits = self.out(z)
        w = self.softmax(logits)  # [B, n_experts]
        return w