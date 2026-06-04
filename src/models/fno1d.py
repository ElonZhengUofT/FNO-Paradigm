import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_channels, n_grid]
        batch_size, _, n_grid = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # [batch, in_channels, n_freq]
        n_freq = x_ft.shape[-1]

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            n_freq,
            dtype=torch.cfloat,
            device=x.device,
        )

        modes = min(self.modes, n_freq)
        weight = torch.view_as_complex(self.weight[:, :, :modes, :])  # [in, out, modes]
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :modes], weight
        )

        out = torch.fft.irfft(out_ft, n=n_grid, dim=-1)  # [batch, out_channels, n_grid]
        return out


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, width, n_grid]
        return self.spectral(x) + self.pointwise(x)


class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        modes: int,
        width: int,
        layers: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(layers)])
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_grid, in_channels]
        x = self.fc0(x)  # [batch, n_grid, width]
        x = x.permute(0, 2, 1)  # [batch, width, n_grid]

        for block in self.blocks:
            x = F.gelu(block(x))  # [batch, width, n_grid]

        x = x.permute(0, 2, 1)  # [batch, n_grid, width]
        x = F.gelu(self.fc1(x))  # [batch, n_grid, hidden_dim]
        x = self.fc2(x)  # [batch, n_grid, 1]
        return x
