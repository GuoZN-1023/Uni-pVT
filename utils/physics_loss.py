# utils/physics_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    数据项：
      - Weighted MSE：对 Z≈1 一带加权的均方误差（主损失）
      - 相对误差项：可选，一般可以先关掉
    物理项：
      - 非负约束
      - 平滑约束
    """

    def __init__(
        self,
        lambda_nonneg: float = 0.10,
        lambda_smooth: float = 0.05,
        lambda_extreme: float = 0.0,   # 控制 Z≈1 区域的权重强度
        lambda_relative: float = 0.0,  # 相对误差项权重（通常设 0）
        extreme_alpha: float = 1.0,    # 峰值放大系数
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_nonneg = float(lambda_nonneg)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_extreme = float(lambda_extreme)
        self.lambda_relative = float(lambda_relative)
        self.extreme_alpha = float(extreme_alpha)
        self.eps = eps

        # Z≈1 的“中心”和带宽，可以根据你数据的分布微调
        self.center_Z0 = 1.0
        self.center_width = 0.03  # 1±0.03 这一窄带会被重点“照顾”

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # 展平成一维
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # 先算一个普通 MSE 只是用来做日志观察，不参与 total_loss
        mse_base = F.mse_loss(y_pred, y_true)

        # 非负约束
        nonneg_penalty = torch.relu(-y_pred).mean()

        # 平滑约束：同一 batch 内相邻预测不要乱抖
        if y_pred.numel() > 1:
            smooth_penalty = (y_pred[1:] - y_pred[:-1]).pow(2).mean()
        else:
            smooth_penalty = torch.zeros(1, device=y_pred.device)

        # Z≈1 区域加权的 weighted MSE —— 这是主数据项
        diff2 = (y_pred - y_true).pow(2)

        if self.lambda_extreme > 0.0:
            # 距离 Z0 的归一化距离
            dist = (y_true - self.center_Z0) / self.center_width
            # 高斯型权重：Z≈1 时 dist≈0，exp(-0)=1
            peak = self.lambda_extreme * self.extreme_alpha
            w_center = 1.0 + peak * torch.exp(-dist.pow(2))
            # 再做一层截断，避免权重过大
            w_center = torch.clamp(w_center, max=1.0 + peak)
        else:
            w_center = torch.ones_like(diff2)

        weighted_mse = (w_center * diff2).mean()

        # 相对误差项：如果不需要，lambda_relative 设为 0 即可
        rel_loss = torch.zeros(1, device=y_pred.device)
        if self.lambda_relative > 0.0:
            denom = y_true.abs().clamp(min=0.05)
            rel_err = (y_pred - y_true) / denom
            rel_loss = (rel_err.pow(2)).mean()

        total_loss = (
            weighted_mse
            + self.lambda_nonneg * nonneg_penalty
            + self.lambda_smooth * smooth_penalty
            + self.lambda_relative * rel_loss
        )

        # 注意：为了兼容 trainer.py，这里仍然给出 "ExtremeMSE" 键，
        # 现在表示的就是这条加权后的数据项
        loss_dict = {
            "MSE": float(mse_base.detach().cpu()),
            "WeightedMSE": float(weighted_mse.detach().cpu()),
            "ExtremeMSE": float(weighted_mse.detach().cpu()),
            "NonNeg": float(nonneg_penalty.detach().cpu()),
            "Smooth": float(smooth_penalty.detach().cpu()),
            "RelLoss": float(rel_loss.detach().cpu()),
        }

        return total_loss, loss_dict
