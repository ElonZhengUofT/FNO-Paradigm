from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_burgers_mat(path: str | Path, input_key: str = "a", output_key: str = "u") -> dict[str, np.ndarray]:
    """Load Burgers .mat dataset with expected keys similar to FNO datasets."""
    data = loadmat(path)
    if input_key not in data or output_key not in data:
        raise KeyError(
            f"Missing keys in mat file. Expected input='{input_key}' and output='{output_key}'."
        )

    u0 = np.asarray(data[input_key], dtype=np.float32)
    uT = np.asarray(data[output_key], dtype=np.float32)

    if u0.ndim != 2 or uT.ndim != 2:
        raise ValueError("Expected 2D arrays [num_samples, n_grid] for both input and output.")

    n_grid = u0.shape[1]
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False, dtype=np.float32)

    return {"x": x, "u0": u0, "uT": uT}
