from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ansatz.heat_ansatz import heat_ansatz_numpy


def sample_smooth_initial_conditions(
    num_samples: int,
    n_grid: int,
    seed: int,
    max_mode: int = 8,
    amplitude_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=np.float64)
    u0 = np.zeros((num_samples, n_grid), dtype=np.float64)

    for i in range(num_samples):
        coeff_sin = rng.normal(scale=amplitude_scale, size=max_mode)
        coeff_cos = rng.normal(scale=amplitude_scale, size=max_mode)
        sample = np.zeros(n_grid, dtype=np.float64)
        for k in range(1, max_mode + 1):
            weight = 1.0 / (k**2)
            sample += weight * (
                coeff_sin[k - 1] * np.sin(2.0 * np.pi * k * x)
                + coeff_cos[k - 1] * np.cos(2.0 * np.pi * k * x)
            )
        u0[i] = sample

    std = np.std(u0, axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    u0 = u0 / std
    return x.astype(np.float32), u0.astype(np.float32)


def burgers_rhs_spectral(u: np.ndarray, nu: float, k: np.ndarray, dealias_mask: np.ndarray) -> np.ndarray:
    u_hat = np.fft.fft(u)
    nonlinear_hat = -0.5j * k * np.fft.fft(u * u)
    diffusion_hat = -nu * (k**2) * u_hat
    rhs_hat = (nonlinear_hat + diffusion_hat) * dealias_mask
    return np.fft.ifft(rhs_hat).real


def solve_burgers_rk4(u0: np.ndarray, nu: float, T: float, n_steps: int) -> np.ndarray:
    n_grid = u0.shape[-1]
    dt = T / n_steps
    k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    k_cutoff = (2.0 / 3.0) * np.max(np.abs(k))
    dealias_mask = (np.abs(k) <= k_cutoff).astype(np.float64)
    u = u0.astype(np.float64, copy=True)

    for _ in range(n_steps):
        k1 = burgers_rhs_spectral(u, nu, k, dealias_mask)
        k2 = burgers_rhs_spectral(u + 0.5 * dt * k1, nu, k, dealias_mask)
        k3 = burgers_rhs_spectral(u + 0.5 * dt * k2, nu, k, dealias_mask)
        k4 = burgers_rhs_spectral(u + dt * k3, nu, k, dealias_mask)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return u.astype(np.float32)


def generate_burgers_dataset(
    num_samples: int,
    n_grid: int,
    nu: float,
    T: float,
    seed: int,
    n_steps: int | None = None,
) -> dict[str, np.ndarray]:
    if n_steps is None:
        n_steps = max(200, int(1000 * T))

    x, u0 = sample_smooth_initial_conditions(num_samples, n_grid, seed)
    uT = np.stack([solve_burgers_rk4(u0_i, nu, T, n_steps) for u0_i in u0], axis=0)
    u_heat = heat_ansatz_numpy(u0, nu=nu, T=T)

    return {
        "x": x,
        "u0": u0.astype(np.float32),
        "uT": uT.astype(np.float32),
        "u_heat": u_heat.astype(np.float32),
    }


def save_burgers_dataset(path: str | Path, data: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
