from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.ansatz.heat_ansatz import heat_ansatz
from src.data.burgers_solver import generate_burgers_dataset, save_burgers_npz
from src.data.mat_reader import read_burgers_mat


@dataclass
class DatasetConfig:
    data_path: str
    num_samples: int
    n_grid: int
    nu: float
    T: float
    seed: int
    train_split: float
    batch_size: int


class BurgersDataset(Dataset):
    def __init__(self, u0: np.ndarray, uT: np.ndarray, u_heat: np.ndarray, x: np.ndarray):
        self.u0 = torch.tensor(u0, dtype=torch.float32)
        self.uT = torch.tensor(uT, dtype=torch.float32)
        self.u_heat = torch.tensor(u_heat, dtype=torch.float32)
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "u0": self.u0[idx],
            "uT": self.uT[idx],
            "u_heat": self.u_heat[idx],
            "x": self.x,
        }


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(path)
    return npz["u0"], npz["uT"], npz["u_heat"], npz["x"]


def ensure_dataset(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cfg.data_path.endswith(".mat"):
        mat = read_burgers_mat(cfg.data_path)
        u0, uT, x = mat["u0"], mat["uT"], mat["x"]
        u_heat = heat_ansatz(u0, nu=cfg.nu, T=cfg.T).astype(np.float32)
        return u0.astype(np.float32), uT.astype(np.float32), u_heat, x.astype(np.float32)

    path = Path(cfg.data_path)
    if path.exists():
        return _load_npz(cfg.data_path)

    u0, uT, u_heat, x = generate_burgers_dataset(
        num_samples=cfg.num_samples,
        n_grid=cfg.n_grid,
        nu=cfg.nu,
        T=cfg.T,
        seed=cfg.seed,
    )
    save_burgers_npz(cfg.data_path, u0, uT, u_heat, x)
    return u0, uT, u_heat, x


def make_dataloaders(cfg: DatasetConfig) -> Tuple[DataLoader, DataLoader]:
    u0, uT, u_heat, x = ensure_dataset(cfg)
    dataset = BurgersDataset(u0=u0, uT=uT, u_heat=u_heat, x=x)

    n_train = int(len(dataset) * cfg.train_split)
    n_test = len(dataset) - n_train
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, test_loader
