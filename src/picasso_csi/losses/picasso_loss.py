"""Composite PICASSO losses for smoke-level model integration."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from picasso_csi.losses.gan_losses import generator_bce_loss
from picasso_csi.losses.physics_losses import (
    delay_sparsity_loss,
    frequency_smoothness_loss,
    pilot_consistency_loss,
)


def reconstruction_loss(
    H_hat: torch.Tensor,
    H_true: torch.Tensor,
    mode: str = "l1",
) -> torch.Tensor:
    """Return scalar reconstruction loss."""

    _validate_same_shape(H_hat, H_true)
    if mode == "l1":
        return F.l1_loss(H_hat, H_true)
    if mode == "mse":
        return F.mse_loss(H_hat, H_true)
    raise ValueError(f"Unsupported reconstruction loss mode {mode!r}.")


def picasso_generator_loss(
    H_hat: torch.Tensor,
    H_true: torch.Tensor,
    H_sparse: torch.Tensor,
    mask: torch.Tensor,
    fake_logits: torch.Tensor | None = None,
    lambda_rec: float = 1.0,
    lambda_adv: float = 0.01,
    lambda_pilot: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_sparse: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return weighted PICASSO generator loss components."""

    _validate_same_shape(H_hat, H_true)
    _validate_same_shape(H_hat, H_sparse)
    _validate_same_shape(H_hat, mask)
    rec_loss = reconstruction_loss(H_hat, H_true, mode="l1")
    if fake_logits is None:
        adv_loss = H_hat.new_zeros(())
    else:
        adv_loss = generator_bce_loss(fake_logits)

    H_hat_physics = _to_channel_last(H_hat)
    H_sparse_physics = _to_channel_last(H_sparse)
    mask_physics = _to_channel_last(mask)
    pilot_loss = pilot_consistency_loss(H_hat_physics, H_sparse_physics, mask_physics)
    smooth_loss = frequency_smoothness_loss(H_hat_physics)
    sparse_loss = delay_sparsity_loss(H_hat_physics)

    total_loss = (
        lambda_rec * rec_loss
        + lambda_adv * adv_loss
        + lambda_pilot * pilot_loss
        + lambda_smooth * smooth_loss
        + lambda_sparse * sparse_loss
    )
    return {
        "total": total_loss,
        "rec": rec_loss,
        "adv": adv_loss,
        "pilot": pilot_loss,
        "smooth": smooth_loss,
        "sparse": sparse_loss,
    }


def _to_channel_last(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 5:
        return tensor.permute(0, 2, 3, 4, 1).contiguous()
    if tensor.ndim == 4:
        return tensor.permute(1, 2, 3, 0).contiguous()
    raise ValueError(
        "CSI tensors must have shape (B, 2, n_rx, n_tx, n_subcarriers) "
        "or (2, n_rx, n_tx, n_subcarriers)."
    )


def _validate_same_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("PICASSO losses expect torch.Tensor inputs.")
    if lhs.shape != rhs.shape:
        raise ValueError(f"Tensor shapes must match, got {lhs.shape} and {rhs.shape}.")
    if lhs.shape[-4] != 2:
        raise ValueError("CSI tensors must use two real-imaginary channels.")
