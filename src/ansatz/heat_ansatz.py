from __future__ import annotations

import numpy as np
import torch


_TWO_PI = 2.0 * np.pi


def _k_squared_numpy(n_grid: int) -> np.ndarray:
    k = _TWO_PI * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    return k**2


def heat_ansatz_numpy(u0: np.ndarray, nu: float, T: float) -> np.ndarray:
    """Compute u_heat(x,T)=exp(nu*T*d_xx)u0 for periodic grid data."""
    n_grid = u0.shape[-1]
    k2 = _k_squared_numpy(n_grid)
    decay = np.exp(-nu * k2 * T)
    u0_hat = np.fft.fft(u0, axis=-1)
    u_heat_hat = u0_hat * decay
    return np.fft.ifft(u_heat_hat, axis=-1).real.astype(np.float32)


def heat_ansatz_torch(u0: torch.Tensor, nu: float, T: float) -> torch.Tensor:
    """Torch variant of heat ansatz for tensor inputs [batch, n_grid]."""
    n_grid = u0.shape[-1]
    k = 2.0 * torch.pi * torch.fft.fftfreq(n_grid, d=1.0 / n_grid, device=u0.device)
    decay = torch.exp(-nu * (k**2) * T)
    u0_hat = torch.fft.fft(u0, dim=-1)
    return torch.fft.ifft(u0_hat * decay, dim=-1).real
