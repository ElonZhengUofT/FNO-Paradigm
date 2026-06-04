from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ansatz.heat_ansatz import heat_ansatz_numpy
from src.data.mat_reader import load_burgers_mat


class BurgersOperatorDataset(Dataset):
    def __init__(
        self,
        u0: np.ndarray,
        uT: np.ndarray,
        x: np.ndarray,
        experiment: str,
        u_heat: np.ndarray | None = None,
    ) -> None:
        self.u0 = torch.from_numpy(u0.astype(np.float32))
        self.uT = torch.from_numpy(uT.astype(np.float32))
        self.x = torch.from_numpy(x.astype(np.float32))
        self.experiment = experiment

        if u_heat is None:
            u_heat = np.zeros_like(uT, dtype=np.float32)
        self.u_heat = torch.from_numpy(u_heat.astype(np.float32))

        self.x_channel = self.x.unsqueeze(0).expand(self.u0.shape[0], -1)

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u0 = self.u0[idx]
        uT = self.uT[idx]
        u_heat = self.u_heat[idx]

        if self.experiment == "baseline":
            features = torch.stack([u0, self.x_channel[idx]], dim=-1)
            target = uT
        elif self.experiment == "residual":
            features = torch.stack([u0, u_heat, self.x_channel[idx]], dim=-1)
            target = uT - u_heat
        else:
            raise ValueError(f"Unsupported experiment: {self.experiment}")

        return {
            "input": features,
            "target": target,
            "u_true": uT,
            "u_heat": u_heat,
        }


def load_or_generate_burgers_data(
    data_path: str,
    num_samples: int,
    n_grid: int,
    nu: float,
    T: float,
    seed: int,
    use_mat: bool = False,
) -> dict[str, np.ndarray]:
    from src.data.burgers_solver import generate_burgers_dataset, save_burgers_dataset

    path = Path(data_path)

    if use_mat:
        loaded = load_burgers_mat(path)
        if "u_heat" not in loaded:
            loaded["u_heat"] = heat_ansatz_numpy(loaded["u0"], nu=nu, T=T)
        return loaded

    if not path.exists():
        generated = generate_burgers_dataset(
            num_samples=num_samples,
            n_grid=n_grid,
            nu=nu,
            T=T,
            seed=seed,
        )
        save_burgers_dataset(path, generated)

    data = np.load(path)
    loaded = {
        "x": data["x"].astype(np.float32),
        "u0": data["u0"].astype(np.float32),
        "uT": data["uT"].astype(np.float32),
    }
    if "u_heat" in data:
        loaded["u_heat"] = data["u_heat"].astype(np.float32)
    else:
        loaded["u_heat"] = heat_ansatz_numpy(loaded["u0"], nu=nu, T=T)
    return loaded


def train_test_split(
    data: dict[str, np.ndarray], test_ratio: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = data["u0"].shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * test_ratio)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train = {}
    test = {}
    for k, v in data.items():
        if k == "x":
            train[k] = v
            test[k] = v
        else:
            train[k] = v[train_idx]
            test[k] = v[test_idx]
    return train, test
