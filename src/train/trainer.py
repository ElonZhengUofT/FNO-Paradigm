from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train.losses import relative_l2_error


@dataclass
class EpochMetrics:
    train_mse: float
    test_mse: float
    test_rel_l2: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        experiment: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.experiment = experiment
        self.criterion = nn.MSELoss()

    def _forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch["input"].to(self.device)
        target = batch["target"].to(self.device)
        u_true = batch["u_true"].to(self.device)
        u_heat = batch["u_heat"].to(self.device)

        if self.experiment == "baseline":
            pred = self.model(features).squeeze(-1)
            supervised_target = target
        else:
            pred, residual = self.model(features, u_heat)
            supervised_target = target
            target = residual * 0.0 + supervised_target

        return pred, supervised_target, u_true

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for batch in loader:
            features = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            u_heat = batch["u_heat"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.experiment == "baseline":
                pred = self.model(features).squeeze(-1)
                loss = self.criterion(pred, target)
            else:
                _, residual = self.model(features, u_heat)
                loss = self.criterion(residual, target)

            loss.backward()
            self.optimizer.step()

            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

        return total_loss / max(total_count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_mse = 0.0
        total_rel_l2 = 0.0
        total_count = 0

        for batch in loader:
            features = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            u_true = batch["u_true"].to(self.device)
            u_heat = batch["u_heat"].to(self.device)

            if self.experiment == "baseline":
                pred = self.model(features).squeeze(-1)
                mse = self.criterion(pred, target)
            else:
                pred, residual = self.model(features, u_heat)
                mse = self.criterion(residual, target)

            rel = relative_l2_error(pred, u_true)

            batch_size = target.shape[0]
            total_mse += mse.item() * batch_size
            total_rel_l2 += rel.item() * batch_size
            total_count += batch_size

        return total_mse / max(total_count, 1), total_rel_l2 / max(total_count, 1)
