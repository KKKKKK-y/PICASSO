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


def delay_domain_sparsity_score(H_hat: torch.Tensor, top_k: int = 6) -> torch.Tensor:
    """Return delay-domain energy concentration in the strongest taps.

    Higher values indicate that more energy is concentrated in a small number
    of delay taps. Inputs use channel-first real-imaginary format with optional
    batch dimension.
    """

    _validate_real_imag_tensor(H_hat)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k!r}.")

    channel_complex = _to_complex(H_hat)
    delay_domain = torch.fft.ifft(channel_complex, dim=-1)
    energy = torch.abs(delay_domain).pow(2)
    n_taps = min(top_k, energy.shape[-1])
    top_energy = torch.topk(energy, k=n_taps, dim=-1).values.sum(dim=-1)
    total_energy = energy.sum(dim=-1).clamp_min(_eps(energy))
    return torch.mean(top_energy / total_energy)


def doppler_robustness_metric(
    H_hat: torch.Tensor,
    H_full: torch.Tensor,
    H_prev: torch.Tensor,
) -> torch.Tensor:
    """Return reconstruction error normalized by meaningful channel energy.

    Temporal variation can be exactly zero for static channels, so the metric
    falls back to the current channel energy in that case.
    """

    _validate_same_shape(H_hat, H_full)
    _validate_same_shape(H_full, H_prev)
    error = torch.sum((H_hat - H_full).pow(2))
    temporal_variation = torch.sum((H_full - H_prev).pow(2))
    channel_energy = torch.sum(H_full.pow(2))
    denominator = torch.maximum(temporal_variation, channel_energy).clamp_min(_eps(H_full))
    return error / denominator


def _validate_same_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("Metrics expect torch.Tensor inputs.")
    if lhs.shape != rhs.shape:
        raise ValueError(f"Tensor shapes must match, got {lhs.shape} and {rhs.shape}.")


def _validate_real_imag_tensor(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Metrics expect torch.Tensor inputs.")
    if tensor.ndim not in (4, 5):
        raise ValueError(
            "CSI tensors must have shape (B, 2, n_rx, n_tx, n_subcarriers) "
            "or (2, n_rx, n_tx, n_subcarriers)."
        )
    if tensor.shape[-4] != 2:
        raise ValueError("CSI tensors must use two real-imaginary channels.")


def _to_complex(tensor: torch.Tensor) -> torch.Tensor:
    _validate_real_imag_tensor(tensor)
    if tensor.ndim == 4:
        return torch.complex(tensor[0], tensor[1])
    return torch.complex(tensor[:, 0], tensor[:, 1])


def _eps(tensor: torch.Tensor) -> float:
    if tensor.dtype.is_floating_point:
        return torch.finfo(tensor.dtype).eps
    return 1e-12
