"""Tests for PICASSO adversarial and composite losses."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.losses import (  # noqa: E402
    discriminator_bce_loss,
    generator_bce_loss,
    picasso_generator_loss,
    reconstruction_loss,
)


def test_gan_losses_return_finite_scalars() -> None:
    real_logits = torch.randn(4, 1)
    fake_logits = torch.randn(4, 1)

    losses = [
        discriminator_bce_loss(real_logits, fake_logits),
        generator_bce_loss(fake_logits),
    ]

    for loss in losses:
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_reconstruction_and_picasso_generator_loss() -> None:
    H_true = torch.randn(4, 2, 4, 4, 64)
    H_sparse = H_true * 0.25
    mask = torch.zeros_like(H_true)
    mask[..., ::4] = 1.0
    H_hat = H_sparse + 0.1 * torch.randn_like(H_sparse)
    fake_logits = torch.randn(4, 1)

    rec = reconstruction_loss(H_hat, H_true)
    losses = picasso_generator_loss(H_hat, H_true, H_sparse, mask, fake_logits=fake_logits)

    assert isinstance(rec, torch.Tensor)
    assert rec.ndim == 0
    assert torch.isfinite(rec)
    assert {"total", "rec", "adv", "pilot", "smooth", "sparse"}.issubset(losses)
    for value in losses.values():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert torch.isfinite(value)
