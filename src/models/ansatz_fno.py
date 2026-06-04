import torch
import torch.nn as nn

from src.models.fno1d import FNO1d


class HeatResidualFNO1d(nn.Module):
    def __init__(self, modes: int, width: int, layers: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = FNO1d(
            in_channels=3,
            modes=modes,
            width=width,
            layers=layers,
            hidden_dim=hidden_dim,
        )

    def forward(self, features: torch.Tensor, u_heat: torch.Tensor):
        # features: [batch, n_grid, 3] = [u0, u_heat, x]
        # u_heat: [batch, n_grid]
        residual = self.backbone(features).squeeze(-1)  # [batch, n_grid]
        pred = residual + u_heat  # [batch, n_grid]
        return residual, pred
