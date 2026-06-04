from pathlib import Path
from typing import Tuple

import numpy as np

from src.ansatz.heat_ansatz import heat_ansatz

SMOOTHNESS_SIGMA = 8.0
SMOOTHNESS_STRENGTH = 0.5


def _dealias_mask(n_grid: int) -> np.ndarray:
    n_modes = n_grid // 2 + 1
    cutoff = int((2.0 / 3.0) * n_modes)
    mask = np.zeros(n_modes, dtype=np.float64)
    mask[:cutoff] = 1.0
    return mask


def _burgers_rhs(u: np.ndarray, nu: float) -> np.ndarray:
    n_grid = u.shape[-1]
    k = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=1.0 / n_grid)
    u_hat = np.fft.rfft(u)

    ux = np.fft.irfft(1j * k * u_hat, n=n_grid)
    nonlinear_hat = np.fft.rfft(u * ux)
    nonlinear_hat *= _dealias_mask(n_grid)

    rhs_hat = -nonlinear_hat - nu * (k**2) * u_hat
    return np.fft.irfft(rhs_hat, n=n_grid)


def solve_burgers_rk4(u0: np.ndarray, nu: float, T: float, dt: float) -> np.ndarray:
    u = u0.copy()
    n_steps = int(np.ceil(T / dt))
    dt_eff = T / n_steps

    for _ in range(n_steps):
        k1 = _burgers_rhs(u, nu)
        k2 = _burgers_rhs(u + 0.5 * dt_eff * k1, nu)
        k3 = _burgers_rhs(u + 0.5 * dt_eff * k2, nu)
        k4 = _burgers_rhs(u + dt_eff * k3, nu)
        u = u + (dt_eff / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def sample_random_smooth_u0(n_grid: int, rng: np.random.Generator) -> np.ndarray:
    n_modes = n_grid // 2 + 1
    k = np.arange(n_modes)
    # Gaussian spectral decay to favor smooth low-frequency initial conditions.
    decay = np.exp(-SMOOTHNESS_STRENGTH * (k / SMOOTHNESS_SIGMA) ** 2)

    real = rng.normal(size=n_modes)
    imag = rng.normal(size=n_modes)
    coeff = (real + 1j * imag) * decay
    coeff[0] = 0.0

    u0 = np.fft.irfft(coeff, n=n_grid)
    u0 = u0 / (np.std(u0) + 1e-8)
    return u0.astype(np.float32)


def generate_burgers_dataset(
    num_samples: int,
    n_grid: int,
    nu: float,
    T: float,
    seed: int,
    dt: float = 5e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=np.float32)

    u0_list, uT_list, uheat_list = [], [], []
    for _ in range(num_samples):
        u0 = sample_random_smooth_u0(n_grid, rng)
        uT = solve_burgers_rk4(u0, nu=nu, T=T, dt=dt).astype(np.float32)
        u_heat = heat_ansatz(u0, nu=nu, T=T).astype(np.float32)
        u0_list.append(u0)
        uT_list.append(uT)
        uheat_list.append(u_heat)

    return (
        np.stack(u0_list, axis=0),
        np.stack(uT_list, axis=0),
        np.stack(uheat_list, axis=0),
        x,
    )


def save_burgers_npz(path: str, u0: np.ndarray, uT: np.ndarray, u_heat: np.ndarray, x: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        u0=u0.astype(np.float32),
        uT=uT.astype(np.float32),
        u_heat=u_heat.astype(np.float32),
        x=x.astype(np.float32),
    )
