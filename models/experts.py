import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Small activation factory so we can easily switch activations from config.

    Supported names (case-insensitive):
      - "relu"
      - "gelu"
      - "tanh"
      - "silu" / "swish"
      - "elu"
      - "leaky_relu" / "lrelu"
      - "selu"
      - "mish"
      - "softplus"
      - "identity" / "linear"
    """
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(0.01)
    if name == "selu":
        return nn.SELU()
    if name == "mish":
        return nn.Mish()
    if name == "softplus":
        return nn.Softplus()
    if name in ("identity", "linear", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    """
    单层全连接残差块：
      y = Dropout( Act( BN( W x + b ) ) ) + Proj(x)

    其中 Proj(x) 在输入输出维度不同的时候使用线性投影，
    相同的时候就是恒等映射。
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        # 残差投影：如果维度不一致，用线性层拉到同一维度
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主分支
        out = self.linear(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        # 残差分支
        res = self.proj(x)

        return out + res


class ExpertNetwork(nn.Module):
    """
    专家网络：若干残差全连接层 + 最后一层线性输出标量。
    每一层都包含：
      Linear -> BatchNorm1d -> Activation -> Dropout -> Residual
    """
    def __init__(self, input_dim, hidden_layers, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size.")

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

        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.out(x)
        return x