import argparse
from typing import Any, Dict

import yaml


def parse_burgers_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
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


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merged_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        cfg[key.replace("-", "_")] = value
    return cfg
