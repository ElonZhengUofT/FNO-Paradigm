from __future__ import annotations

import torch


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diff = torch.norm(pred - target, dim=1)
    denom = torch.norm(target, dim=1).clamp_min(eps)
    return (diff / denom).mean()
