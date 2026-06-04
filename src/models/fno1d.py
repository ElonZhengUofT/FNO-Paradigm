from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D spectral convolution using low-frequency Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels, n_grid]
        batch_size, _, n_grid = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        # x_ft: [batch, in_channels, n_grid//2 + 1]

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )

        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :m], self.weights[:, :, :m]
        )

        # out: [batch, out_channels, n_grid]
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)


class FNOBlock1d(nn.Module):
    """One FNO block: spectral convolution + pointwise bypass."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, width, n_grid]
        y = self.spectral(x) + self.pointwise(x)
        return F.gelu(y)


class FNO1d(nn.Module):
    """Minimal 1D Fourier Neural Operator for operator learning on grids."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        projection_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_lifting = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(n_layers)])
        self.output_projection = nn.Sequential(
            nn.Linear(width, projection_hidden_dim),
            nn.GELU(),
            nn.Linear(projection_hidden_dim, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_grid, in_channels]
        x = self.input_lifting(x)
        # x: [batch, n_grid, width]
        x = x.permute(0, 2, 1)
        # x: [batch, width, n_grid]

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 1)
        # x: [batch, n_grid, width]
        x = self.output_projection(x)
        # x: [batch, n_grid, out_channels]
        return x
