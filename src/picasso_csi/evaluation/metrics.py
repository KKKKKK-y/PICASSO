"""Evaluation metrics for CSI reconstruction."""

from __future__ import annotations

import torch


def mse(H_hat: torch.Tensor, H_full: torch.Tensor) -> torch.Tensor:
    """Return mean squared error as a scalar tensor."""

    _validate_same_shape(H_hat, H_full)
    return torch.mean((H_hat - H_full).pow(2))


def mae(H_hat: torch.Tensor, H_full: torch.Tensor) -> torch.Tensor:
    """Return mean absolute error as a scalar tensor."""

    _validate_same_shape(H_hat, H_full)
    return torch.mean(torch.abs(H_hat - H_full))


def nmse(H_hat: torch.Tensor, H_full: torch.Tensor) -> torch.Tensor:
    """Return normalized mean squared error as ``||e||_2^2 / ||H||_2^2``."""

    _validate_same_shape(H_hat, H_full)
    numerator = torch.sum((H_hat - H_full).pow(2))
    denominator = torch.sum(H_full.pow(2)).clamp_min(_eps(H_full))
    return numerator / denominator


def pilot_consistency_error(
    H_hat: torch.Tensor,
    H_sparse: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return observed-pilot MSE as a scalar tensor."""

    _validate_same_shape(H_hat, H_sparse)
    _validate_same_shape(H_hat, mask)
    numerator = torch.sum(((H_hat - H_sparse) * mask).pow(2))
    denominator = torch.sum(mask).clamp_min(_eps(mask))
    return numerator / denominator


def _validate_same_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("Metrics expect torch.Tensor inputs.")
    if lhs.shape != rhs.shape:
        raise ValueError(f"Tensor shapes must match, got {lhs.shape} and {rhs.shape}.")


def _eps(tensor: torch.Tensor) -> float:
    if tensor.dtype.is_floating_point:
        return torch.finfo(tensor.dtype).eps
    return 1e-12
