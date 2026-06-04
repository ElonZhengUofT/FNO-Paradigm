import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data.burgers_dataset import DatasetConfig, make_dataloaders
from src.models.ansatz_fno import HeatResidualFNO1d
from src.models.fno1d import FNO1d
from src.train.losses import mse_loss, relative_l2_error
from src.utils.checkpoint import save_checkpoint
from src.utils.plotting import save_comparison_plot

ExperimentType = Literal["baseline", "residual"]


@dataclass
class TrainConfig:
    experiment: ExperimentType
    n_grid: int
    num_samples: int
    epochs: int
    batch_size: int
    modes: int
    width: int
    layers: int
    nu: float
    T: float
    lr: float
    seed: int
    data_path: str
    save_dir: str
    train_split: float = 0.8
    plot_samples: int = 3
    hidden_dim: int = 128


def _make_features(batch: Dict[str, torch.Tensor], experiment: ExperimentType) -> torch.Tensor:
    u0 = batch["u0"]
    u_heat = batch["u_heat"]
    x = batch["x"].unsqueeze(0).repeat(u0.shape[0], 1)

    if experiment == "baseline":
        return torch.stack([u0, x], dim=-1)
    return torch.stack([u0, u_heat, x], dim=-1)


def _build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.experiment == "baseline":
        return FNO1d(
            in_channels=2,
            modes=cfg.modes,
            width=cfg.width,
            layers=cfg.layers,
            hidden_dim=cfg.hidden_dim,
        )
    return HeatResidualFNO1d(
        modes=cfg.modes,
        width=cfg.width,
        layers=cfg.layers,
        hidden_dim=cfg.hidden_dim,
    )


def _step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    experiment: ExperimentType,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) for k, v in batch.items()}
    features = _make_features(batch, experiment)
    u_true = batch["uT"]

    if experiment == "baseline":
        pred = model(features).squeeze(-1)
        loss = mse_loss(pred, u_true)
        return loss, pred, u_true

    residual_true = u_true - batch["u_heat"]
    residual_pred, pred = model(features, batch["u_heat"])
    loss = mse_loss(residual_pred, residual_true)
    return loss, pred, u_true


def run_training(cfg: TrainConfig) -> Dict[str, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = DatasetConfig(
        data_path=cfg.data_path,
        num_samples=cfg.num_samples,
        n_grid=cfg.n_grid,
        nu=cfg.nu,
        T=cfg.T,
        seed=cfg.seed,
        train_split=cfg.train_split,
        batch_size=cfg.batch_size,
    )
    train_loader, test_loader = make_dataloaders(data_cfg)

    model = _build_model(cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    save_root = Path(cfg.save_dir) / cfg.experiment
    save_root.mkdir(parents=True, exist_ok=True)

    metrics = {"train_mse": [], "test_mse": [], "test_rel_l2": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _, _ = _step(model, batch, cfg.experiment, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        test_mse_vals = []
        test_rel_vals = []
        with torch.no_grad():
            for batch in test_loader:
                loss, pred, true = _step(model, batch, cfg.experiment, device)
                test_mse_vals.append(loss.item())
                test_rel_vals.append(relative_l2_error(pred, true).item())

        train_mse = float(np.mean(train_losses))
        test_mse = float(np.mean(test_mse_vals))
        test_rel = float(np.mean(test_rel_vals))

        metrics["train_mse"].append(train_mse)
        metrics["test_mse"].append(test_mse)
        metrics["test_rel_l2"].append(test_rel)

        print(
            f"[{cfg.experiment}] epoch={epoch:04d} train_mse={train_mse:.6e} "
            f"test_mse={test_mse:.6e} test_rel_l2={test_rel:.6e}"
        )

    with (save_root / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with (save_root / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_checkpoint(
        str(save_root / "model.pt"),
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "metrics": metrics,
        },
    )

    _save_sample_plots(model, test_loader, cfg, save_root, device)
    _save_summary_markdown(metrics, cfg, save_root)

    return metrics


def _save_sample_plots(
    model: nn.Module,
    test_loader,
    cfg: TrainConfig,
    save_root: Path,
    device: torch.device,
) -> None:
    model.eval()
    batch = next(iter(test_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    features = _make_features(batch, cfg.experiment)

    with torch.no_grad():
        if cfg.experiment == "baseline":
            pred = model(features).squeeze(-1)
            residual_pred = pred - batch["u_heat"]
        else:
            residual_pred, pred = model(features, batch["u_heat"])

    x = batch["x"][0].detach().cpu().numpy()
    n = min(cfg.plot_samples, pred.shape[0])

    for i in range(n):
        u0 = batch["u0"][i].detach().cpu().numpy()
        u_true = batch["uT"][i].detach().cpu().numpy()
        u_heat = batch["u_heat"][i].detach().cpu().numpy()
        u_pred = pred[i].detach().cpu().numpy()
        res_pred = residual_pred[i].detach().cpu().numpy()
        res_err = (u_true - u_heat) - res_pred

        save_comparison_plot(
            path=str(save_root / f"sample_{i}.png"),
            x=x,
            title=f"{cfg.experiment} sample {i}",
            series={
                "u0": u0,
                "u_true": u_true,
                "u_heat": u_heat,
                "baseline_or_total_pred": u_pred,
                "residual_pred": res_pred,
                "residual_error": res_err,
            },
        )


def _save_summary_markdown(metrics: Dict[str, List[float]], cfg: TrainConfig, save_root: Path) -> None:
    final_train = metrics["train_mse"][-1]
    final_test = metrics["test_mse"][-1]
    final_rel = metrics["test_rel_l2"][-1]

    text = (
        f"# {cfg.experiment} summary\n\n"
        f"- epochs: {cfg.epochs}\n"
        f"- final train MSE: {final_train:.6e}\n"
        f"- final test MSE: {final_test:.6e}\n"
        f"- final test relative L2: {final_rel:.6e}\n"
    )
    with (save_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write(text)
