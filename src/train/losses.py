import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute per-sample relative L2 errors and return the batch mean.
    pred/target: [batch, grid]
    returns: scalar tensor with batch-averaged relative L2 error
    """
    # pred/target: [batch, grid]
    num = torch.linalg.norm(pred - target, dim=1)
    den = torch.linalg.norm(target, dim=1).clamp_min(eps)
    return torch.mean(num / den)
