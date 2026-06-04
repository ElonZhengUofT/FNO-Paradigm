#!/usr/bin/env python
import argparse
from pathlib import Path

import yaml

from src.train.trainer import TrainConfig, run_training
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and residual FNO on Burgers")
    parser.add_argument("--config", type=str, default="configs/burgers_default.yaml")
    parser.add_argument("--experiment", type=str, choices=["baseline", "residual", "both"], default=None)
    parser.add_argument("--n-grid", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--plot-samples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merged_config(args):
    cfg = load_config(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        cfg[key.replace("-", "_")] = value
    return cfg


def run_single(cfg_dict, experiment_name: str):
    cfg = TrainConfig(
        experiment=experiment_name,
        n_grid=cfg_dict["n_grid"],
        num_samples=cfg_dict["num_samples"],
        epochs=cfg_dict["epochs"],
        batch_size=cfg_dict["batch_size"],
        modes=cfg_dict["modes"],
        width=cfg_dict["width"],
        layers=cfg_dict["layers"],
        nu=cfg_dict["nu"],
        T=cfg_dict["T"],
        lr=cfg_dict["lr"],
        seed=cfg_dict["seed"],
        data_path=cfg_dict["data_path"],
        save_dir=cfg_dict["save_dir"],
        train_split=cfg_dict["train_split"],
        plot_samples=cfg_dict["plot_samples"],
        hidden_dim=cfg_dict["hidden_dim"],
    )
    run_training(cfg)


def main():
    args = parse_args()
    cfg = merged_config(args)
    set_seed(cfg["seed"])

    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    if cfg["experiment"] == "both":
        run_single(cfg, "baseline")
        run_single(cfg, "residual")
    else:
        run_single(cfg, cfg["experiment"])


if __name__ == "__main__":
    main()
