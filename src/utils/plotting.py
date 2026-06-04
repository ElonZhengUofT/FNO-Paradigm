from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_comparison_plot(
    save_path: str | Path,
    x: np.ndarray,
    u0: np.ndarray,
    u_true: np.ndarray,
    u_heat: np.ndarray,
    baseline_pred: np.ndarray | None = None,
    residual_pred: np.ndarray | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(x, u0, label="u0", linewidth=1.5)
    plt.plot(x, u_true, label="u_true", linewidth=2)
    plt.plot(x, u_heat, label="u_heat", linestyle="--")
    if baseline_pred is not None:
        plt.plot(x, baseline_pred, label="baseline_pred", alpha=0.85)
    if residual_pred is not None:
        plt.plot(x, residual_pred, label="residual_pred", alpha=0.85)
        plt.plot(x, residual_pred - u_true, label="residual_error", alpha=0.85)
    plt.xlabel("x")
    plt.ylabel("u(x, T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
