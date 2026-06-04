#!/usr/bin/env python

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_burgers import merged_config, parse_args
from src.train.trainer import TrainConfig, run_training
from src.utils.seed import set_seed


def main():
    args = parse_args()
    cfg = merged_config(args)
    cfg["experiment"] = "baseline"
    set_seed(cfg["seed"])

    run_training(
        TrainConfig(
            experiment="baseline",
            n_grid=cfg["n_grid"],
            num_samples=cfg["num_samples"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            modes=cfg["modes"],
            width=cfg["width"],
            layers=cfg["layers"],
            nu=cfg["nu"],
            T=cfg["T"],
            lr=cfg["lr"],
            seed=cfg["seed"],
            data_path=cfg["data_path"],
            save_dir=cfg["save_dir"],
            train_split=cfg["train_split"],
            plot_samples=cfg["plot_samples"],
            hidden_dim=cfg["hidden_dim"],
        )
    )


if __name__ == "__main__":
    main()
