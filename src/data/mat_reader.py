from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import loadmat


def read_burgers_mat(path: str) -> Dict[str, np.ndarray]:
    mat = loadmat(path)
    keys = {k.lower(): k for k in mat.keys()}

    def find_key(candidates):
        for c in candidates:
            if c in keys:
                return keys[c]
        return None

    u0_key = find_key(["u0", "a", "input", "u_init"])
    uT_key = find_key(["ut", "u", "output", "u_target"])
    x_key = find_key(["x", "grid"])

    if u0_key is None or uT_key is None:
        raise ValueError(f"Could not infer u0/uT keys in MAT file: {Path(path).name}")

    u0 = np.asarray(mat[u0_key], dtype=np.float32)
    uT = np.asarray(mat[uT_key], dtype=np.float32)
    if u0.ndim == 1:
        u0 = u0[None, :]
    if uT.ndim == 1:
        uT = uT[None, :]

    if x_key is not None:
        x = np.asarray(mat[x_key], dtype=np.float32).reshape(-1)
    else:
        x = np.linspace(0.0, 1.0, u0.shape[-1], endpoint=False, dtype=np.float32)

    return {"u0": u0, "uT": uT, "x": x}
