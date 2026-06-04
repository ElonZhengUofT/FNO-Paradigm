from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.burgers_dataset import (  # noqa: E402
    BurgersOperatorDataset,
    load_or_generate_burgers_data,
    train_test_split,
)
from src.models.ansatz_fno import HeatResidualFNO  # noqa: E402
from src.models.fno1d import FNO1d  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402
from src.utils.checkpoint import save_checkpoint, save_json  # noqa: E402
from src.utils.plotting import save_comparison_plot  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and residual FNO for Burgers")
    parser.add_argument("--config", type=str, default="configs/burgers_default.yaml")
    parser.add_argument("--experiment", type=str, choices=["baseline", "residual", "both"], default=None)
    parser.add_argument("--n-grid", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot-samples", type=int, default=None)
    parser.add_argument("--use-mat", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            cfg[key.replace("-", "_")] = value

    if cfg.get("device", "auto") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def make_dataloaders(cfg: dict, experiment: str):
    raw = load_or_generate_burgers_data(
        data_path=cfg["data_path"],
        num_samples=cfg["num_samples"],
        n_grid=cfg["n_grid"],
        nu=cfg["nu"],
        T=cfg["T"],
        seed=cfg["seed"],
        use_mat=cfg.get("use_mat", False),
    )
    train_data, test_data = train_test_split(raw, test_ratio=cfg["test_ratio"], seed=cfg["seed"])

    train_ds = BurgersOperatorDataset(
        u0=train_data["u0"],
        uT=train_data["uT"],
        x=raw["x"],
        experiment=experiment,
        u_heat=train_data.get("u_heat"),
    )
    test_ds = BurgersOperatorDataset(
        u0=test_data["u0"],
        uT=test_data["uT"],
        x=raw["x"],
        experiment=experiment,
        u_heat=test_data.get("u_heat"),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
    return train_loader, test_loader, raw


def build_model(cfg: dict, experiment: str) -> torch.nn.Module:
    if experiment == "baseline":
        return FNO1d(
            in_channels=2,
            out_channels=1,
            modes=cfg["modes"],
            width=cfg["width"],
            n_layers=cfg["layers"],
            projection_hidden_dim=cfg["hidden_dim"],
        )
    return HeatResidualFNO(
        modes=cfg["modes"],
        width=cfg["width"],
        layers=cfg["layers"],
        hidden_dim=cfg["hidden_dim"],
    )


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, experiment: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, truths, u0s, u_heats = [], [], [], []
    for batch in loader:
        features = batch["input"].to(device)
        u_true = batch["u_true"].to(device)
        u_heat = batch["u_heat"].to(device)
        if experiment == "baseline":
            pred = model(features).squeeze(-1)
        else:
            pred, _ = model(features, u_heat)

        preds.append(pred.cpu().numpy())
        truths.append(u_true.cpu().numpy())
        u0s.append(features[..., 0].cpu().numpy())
        u_heats.append(u_heat.cpu().numpy())

    return (
        np.concatenate(u0s, axis=0),
        np.concatenate(truths, axis=0),
        np.concatenate(u_heats, axis=0),
        np.concatenate(preds, axis=0),
    )


def run_single_experiment(cfg: dict, experiment: str) -> dict:
    save_dir = Path(cfg["save_dir"]) / experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, raw = make_dataloaders(cfg, experiment)

    device = torch.device(cfg["device"])
    model = build_model(cfg, experiment).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    trainer = Trainer(model=model, optimizer=optimizer, device=device, experiment=experiment)

    metrics_history: list[dict[str, float]] = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_mse = trainer.train_epoch(train_loader)
        test_mse, test_rel_l2 = trainer.evaluate(test_loader)
        row = {"epoch": epoch, "train_mse": train_mse, "test_mse": test_mse, "test_rel_l2": test_rel_l2}
        metrics_history.append(row)
        print(
            f"[{experiment}] epoch={epoch:04d} "
            f"train_mse={train_mse:.6e} test_mse={test_mse:.6e} test_rel_l2={test_rel_l2:.6e}"
        )

    save_json(save_dir / "metrics.json", {"history": metrics_history, "final": metrics_history[-1]})
    save_checkpoint(save_dir / "model.pt", model, optimizer, cfg["epochs"])

    u0, u_true, u_heat, pred = collect_predictions(model, test_loader, experiment, device)
    x = raw["x"]
    n_plot = min(cfg["plot_samples"], pred.shape[0])
    for i in range(n_plot):
        baseline_pred = pred[i] if experiment == "baseline" else None
        residual_pred = pred[i] if experiment == "residual" else None
        save_comparison_plot(
            save_path=save_dir / f"comparison_{i:03d}.png",
            x=x,
            u0=u0[i],
            u_true=u_true[i],
            u_heat=u_heat[i],
            baseline_pred=baseline_pred,
            residual_pred=residual_pred,
        )

    summary_md = (
        f"# {experiment} summary\n\n"
        f"- epochs: {cfg['epochs']}\n"
        f"- final train MSE: {metrics_history[-1]['train_mse']:.6e}\n"
        f"- final test MSE: {metrics_history[-1]['test_mse']:.6e}\n"
        f"- final test relative L2: {metrics_history[-1]['test_rel_l2']:.6e}\n"
    )
    (save_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    save_json(save_dir / "config.json", cfg)
    return metrics_history[-1]


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    set_seed(cfg["seed"])

    experiments = [cfg["experiment"]] if cfg["experiment"] != "both" else ["baseline", "residual"]

    results = {}
    for exp in experiments:
        results[exp] = run_single_experiment(cfg, exp)

    final_path = Path(cfg["save_dir"]) / "comparison_metrics.json"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
