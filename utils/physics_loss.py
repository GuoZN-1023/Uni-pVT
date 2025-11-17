import torch
import torch.nn as nn

class PhysicsLoss(nn.Module):
    def __init__(self, lambda_nonneg=0.1, lambda_smooth=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l_nonneg = float(lambda_nonneg)
        self.l_smooth = float(lambda_smooth)

    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)
        nonneg_loss = torch.mean(torch.relu(-preds))
        if preds.size(0) > 1:
            smoothness_loss = torch.mean((preds[1:] - preds[:-1]) ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=preds.device)
        total = mse_loss + self.l_nonneg * nonneg_loss + self.l_smooth * smoothness_loss
        return total, {
            "MSE": float(mse_loss.detach().cpu()),
            "NonNeg": float(nonneg_loss.detach().cpu()),
            "Smooth": float(smoothness_loss.detach().cpu())
        }