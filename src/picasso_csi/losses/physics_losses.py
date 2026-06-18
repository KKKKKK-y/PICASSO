"""Minimal physics-informed loss interfaces for PICASSO Stage 0."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pilot_consistency_loss(
    H_hat: torch.Tensor,
    H_sparse: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared error on observed pilot positions."""

    _validate_real_imag_tensor(H_hat, "H_hat")
    _validate_same_shape(H_hat, H_sparse, "H_hat", "H_sparse")
    _validate_same_shape(H_hat, mask, "H_hat", "mask")
    return F.mse_loss(H_hat * mask, H_sparse * mask)


def frequency_smoothness_loss(H_hat: torch.Tensor) -> torch.Tensor:
    """Penalize abrupt adjacent-subcarrier changes."""

    _validate_real_imag_tensor(H_hat, "H_hat")
    if H_hat.shape[-2] < 2:
        return H_hat.new_zeros(())
    diff = H_hat[..., 1:, :] - H_hat[..., :-1, :]
    return torch.mean(diff.pow(2))


def delay_sparsity_loss(H_hat: torch.Tensor) -> torch.Tensor:
    """L1-style sparsity proxy in the delay domain."""

    _validate_real_imag_tensor(H_hat, "H_hat")
    channel_complex = torch.view_as_complex(H_hat.contiguous())
    delay_domain = torch.fft.ifft(channel_complex, dim=-1)
    return torch.mean(torch.abs(delay_domain))


def frequency_consistency_loss(
    H_hat: torch.Tensor,
    H_sparse: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Observed-subcarrier consistency in the frequency domain."""

    return pilot_consistency_loss(H_hat, H_sparse, mask)


def energy_preservation_loss(
    H_hat: torch.Tensor,
    H_reference: torch.Tensor,
) -> torch.Tensor:
    """Penalize mismatch between average channel energies."""

    _validate_real_imag_tensor(H_hat, "H_hat")
    _validate_same_shape(H_hat, H_reference, "H_hat", "H_reference")
    energy_hat = torch.mean(H_hat.pow(2), dim=(-4, -3, -2, -1))
    energy_ref = torch.mean(H_reference.pow(2), dim=(-4, -3, -2, -1))
    return F.mse_loss(energy_hat, energy_ref)


def _validate_real_imag_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.shape[-1] != 2:
        raise ValueError(f"{name} must use real-imaginary format with final dimension 2.")


def _validate_same_shape(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_name: str,
    rhs_name: str,
) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{lhs_name} and {rhs_name} must have the same shape.")
