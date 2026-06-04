from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def save_comparison_plot(path: str, x: np.ndarray, series: Dict[str, np.ndarray], title: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, y in series.items():
        plt.plot(x, y, label=name)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
