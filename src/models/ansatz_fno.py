from __future__ import annotations

import torch
from torch import nn

from src.models.fno1d import FNO1d


class HeatResidualFNO(nn.Module):
    """Residual FNO model for Burgers: u_pred = u_heat + residual."""

    def __init__(self, modes: int, width: int, layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.residual_net = FNO1d(
            in_channels=3,
            out_channels=1,
            modes=modes,
            width=width,
            n_layers=layers,
            projection_hidden_dim=hidden_dim,
        )

    def forward(self, features: torch.Tensor, u_heat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.residual_net(features).squeeze(-1)
        pred = u_heat + residual
        return pred, residual
