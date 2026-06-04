import numpy as np


def heat_ansatz(u0: np.ndarray, nu: float, T: float) -> np.ndarray:
    """
    u0: [n_grid] or [batch, n_grid]
    returns: same shape as u0
    """
    u0 = np.asarray(u0)
    n_grid = u0.shape[-1]
    k = 2.0 * np.pi * np.fft.rfftfreq(n_grid, d=1.0 / n_grid)
    decay = np.exp(-nu * (k**2) * T)
    u0_hat = np.fft.rfft(u0, axis=-1)
    u_heat_hat = u0_hat * decay
    return np.fft.irfft(u_heat_hat, n=n_grid, axis=-1)
