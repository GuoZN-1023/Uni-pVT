# models/fusion_model.py
import torch
import torch.nn as nn
from .experts import ExpertNetwork
from .gate import GateNetwork


class FusionModel(nn.Module):
    """
    四专家 Mixture-of-Experts 模型：
      - 四个 ExpertNetwork：气相 / 液相 / 临界 / 额外区域
      - 一个 GateNetwork：输出四个权重
      - 前向：每个专家各算一个标量，再按 gate 权重加权求和
    """
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg["model"]["input_dim"]

        gas_cfg   = cfg["experts"]["gas"]
        liq_cfg   = cfg["experts"]["liquid"]
        crit_cfg  = cfg["experts"]["critical"]
        extra_cfg = cfg["experts"]["extra"]
        gate_cfg  = cfg["gate"]

        n_experts = 4

        self.expert_gas = ExpertNetwork(
            input_dim,
            gas_cfg["hidden_layers"],
            gas_cfg["activation"],
            gas_cfg["dropout"],
        )
        self.expert_liq = ExpertNetwork(
            input_dim,
            liq_cfg["hidden_layers"],
            liq_cfg["activation"],
            liq_cfg["dropout"],
        )
        self.expert_crit = ExpertNetwork(
            input_dim,
            crit_cfg["hidden_layers"],
            crit_cfg["activation"],
            crit_cfg["dropout"],
        )
        self.expert_extra = ExpertNetwork(
            input_dim,
            extra_cfg["hidden_layers"],
            extra_cfg["activation"],
            extra_cfg["dropout"],
        )

        self.gate = GateNetwork(
            input_dim,
            gate_cfg["hidden_layers"],
            gate_cfg["activation"],
            gate_cfg["dropout"],
            n_experts=n_experts,
        )

    def forward(self, x):
        # 门控权重
        w = self.gate(x)  # [B,4]

        # 四个专家各自输出
        out_gas   = self.expert_gas(x)      # [B,1]
        out_liq   = self.expert_liq(x)      # [B,1]
        out_crit  = self.expert_crit(x)     # [B,1]
        out_extra = self.expert_extra(x)    # [B,1]

        outputs = torch.cat(
            [out_gas, out_liq, out_crit, out_extra], dim=1
        )  # [B,4]

        fused = torch.sum(w * outputs, dim=1, keepdim=True)  # [B,1]
        return fused, w, outputs
