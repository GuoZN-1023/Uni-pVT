import torch
import torch.nn as nn
from .experts import ExpertNetwork
from .gate import GateNetwork

class FusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg["model"]["input_dim"]
        gas_cfg = cfg["experts"]["gas"]
        liq_cfg = cfg["experts"]["liquid"]
        crit_cfg = cfg["experts"]["critical"]
        gate_cfg = cfg["gate"]

        self.expert_gas = ExpertNetwork(input_dim, gas_cfg["hidden_layers"], gas_cfg["activation"], gas_cfg["dropout"])
        self.expert_liq = ExpertNetwork(input_dim, liq_cfg["hidden_layers"], liq_cfg["activation"], liq_cfg["dropout"])
        self.expert_crit = ExpertNetwork(input_dim, crit_cfg["hidden_layers"], crit_cfg["activation"], crit_cfg["dropout"])
        self.gate = GateNetwork(input_dim, gate_cfg["hidden_layers"], gate_cfg["activation"], gate_cfg["dropout"])

    def forward(self, x):
        w = self.gate(x)  # [B,3]
        out_gas = self.expert_gas(x)   # [B,1]
        out_liq = self.expert_liq(x)   # [B,1]
        out_crit = self.expert_crit(x) # [B,1]
        outputs = torch.cat([out_gas, out_liq, out_crit], dim=1)  # [B,3]
        fused = torch.sum(w * outputs, dim=1, keepdim=True)       # [B,1]
        return fused, w, outputs