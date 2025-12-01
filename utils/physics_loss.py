# utils/physics_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    Loss = region-weighted(data loss) + lambda_nonneg*nonneg + lambda_smooth*smooth + lambda_entropy*entropy(optional)

    - data loss supports: mse / huber
    - region-weighted: expert_id (1..4) -> weight
    - entropy regularization: encourage sharper gate weights (optional)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        tr = cfg.get("training", {})
        self.loss_type = str(tr.get("loss_type", "mse")).lower()   # "mse" or "huber"
        self.huber_delta = float(tr.get("huber_delta", 1.0))

        self.lambda_nonneg = float(tr.get("lambda_nonneg", 0.0))
        self.lambda_smooth = float(tr.get("lambda_smooth", 0.0))
        self.lambda_entropy = float(tr.get("lambda_entropy", 0.0))  # encourage sharp gate (optional)

        # region weights: dict like {1:1.0, 2:1.3, 3:1.6, 4:1.2}
        self.region_weights = tr.get("region_weights", None)
        if isinstance(self.region_weights, dict):
            self.region_weights = {int(k): float(v) for k, v in self.region_weights.items()}
        else:
            self.region_weights = None

    def _data_loss_vec(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return per-sample loss vector shape (B,)"""
        pred = pred.view(-1)
        target = target.view(-1)

        if self.loss_type == "huber":
            # smooth_l1_loss gives per-element if reduction='none'
            return F.smooth_l1_loss(pred, target, beta=self.huber_delta, reduction="none")
        # default mse
        return (pred - target) ** 2

    def _region_weight_vec(self, expert_id: torch.Tensor, device) -> torch.Tensor:
        """expert_id shape (B,), values 1..4"""
        if self.region_weights is None or expert_id is None:
            return None
        eid = expert_id.view(-1).long().to(device)
        w = torch.ones_like(eid, dtype=torch.float32, device=device)
        for k, v in self.region_weights.items():
            w = torch.where(eid == int(k), torch.tensor(float(v), device=device), w)
        return w

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        expert_id: torch.Tensor = None,
        gate_w: torch.Tensor = None,
        extra: dict = None,
    ):
        """
        pred: (B,) or (B,1)
        target: (B,) or (B,1)
        expert_id: (B,) optional
        gate_w: (B,4) optional, for entropy reg
        extra: optional dict for smooth etc.
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # -------- data loss (region weighted) --------
        loss_vec = self._data_loss_vec(pred, target)  # (B,)
        w_reg = self._region_weight_vec(expert_id, pred.device)
        if w_reg is not None:
            loss_data = (w_reg * loss_vec).mean()
        else:
            loss_data = loss_vec.mean()

        # -------- nonneg penalty --------
        loss_nonneg = torch.tensor(0.0, device=pred.device)
        if self.lambda_nonneg > 0:
            loss_nonneg = F.relu(-pred).mean()

        # -------- smooth penalty (keep your original convention; uses batch adjacency) --------
        loss_smooth = torch.tensor(0.0, device=pred.device)
        if self.lambda_smooth > 0:
            if extra is not None and isinstance(extra, dict) and "pred_sorted" in extra:
                ps = extra["pred_sorted"].view(-1)
            else:
                ps = pred
            if ps.numel() >= 2:
                loss_smooth = ((ps[1:] - ps[:-1]) ** 2).mean()

        # -------- entropy penalty (sharpen gate) --------
        # We add: lambda_entropy * mean( sum_i w_i * log(w_i) )   (negative entropy; makes distribution sharper)
        loss_entropy = torch.tensor(0.0, device=pred.device)
        if self.lambda_entropy > 0 and gate_w is not None:
            w = torch.clamp(gate_w, 1e-12, 1.0)
            loss_entropy = (w * torch.log(w)).sum(dim=1).mean()

        total = loss_data + self.lambda_nonneg * loss_nonneg + self.lambda_smooth * loss_smooth + self.lambda_entropy * loss_entropy

        loss_dict = {
            "MSE": float(loss_data.detach().cpu().item()),
            "NonNeg": float(loss_nonneg.detach().cpu().item()),
            "Smooth": float(loss_smooth.detach().cpu().item()),
            "Entropy": float(loss_entropy.detach().cpu().item()),
        }
        return total, loss_dict
