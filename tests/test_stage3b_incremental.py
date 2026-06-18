"""Tests for Stage 3B incremental structural enhancements."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.losses import energy_preservation_loss, frequency_consistency_loss, picasso_generator_loss  # noqa: E402
from picasso_csi.models import PICASSODiscriminator, PICASSOGenerator  # noqa: E402
from picasso_csi.training.run_stage3b_incremental import _method_specs  # noqa: E402


def test_incremental_generator_options_forward_shape() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    model = PICASSOGenerator(
        base_channels=16,
        num_blocks=2,
        refinement_blocks=2,
        use_multiscale_fusion=True,
        use_channel_attention=True,
    )

    H_hat = model(H_sparse)

    assert H_hat.shape == H_sparse.shape
    assert torch.isfinite(H_hat).all()


def test_film_conditioning_forward_shape() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    model = PICASSOGenerator(
        base_channels=16,
        num_blocks=2,
        use_condition=True,
        use_film_conditioning=True,
    )

    H_hat = model(H_sparse, torch.tensor([10.0, 20.0]), torch.tensor([0.125, 0.25]))

    assert H_hat.shape == H_sparse.shape
    assert torch.isfinite(H_hat).all()


def test_delay_feature_discriminator_forward_shape() -> None:
    H = torch.randn(2, 2, 4, 4, 64)
    discriminator = PICASSODiscriminator(base_channels=16, use_delay_features=True)

    logits = discriminator(H)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_enhanced_physics_losses_are_finite() -> None:
    H_true = torch.randn(2, 2, 4, 4, 16)
    H_sparse = H_true * 0.25
    mask = torch.ones_like(H_true)
    H_hat = H_sparse + 0.1 * torch.randn_like(H_sparse)
    channel_last = H_hat.permute(0, 2, 3, 4, 1).contiguous()

    losses = picasso_generator_loss(
        H_hat,
        H_true,
        H_sparse,
        mask,
        lambda_frequency=0.5,
        lambda_energy=0.05,
        loss_mode="rec_physics",
    )

    assert torch.isfinite(frequency_consistency_loss(channel_last, channel_last, torch.ones_like(channel_last)))
    assert torch.isfinite(energy_preservation_loss(channel_last, channel_last))
    assert torch.isfinite(losses["frequency"])
    assert torch.isfinite(losses["energy"])


def test_stage3b_config_and_methods() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "stage3b_incremental.yaml").read_text(encoding="utf-8"))
    methods = [method["name"] for method in _method_specs(config)]

    assert config["project"]["stage"] == "stage3b_incremental"
    assert "PICASSO-rec-base" in methods
    assert "PICASSO-full-feature" in methods
    assert "PICASSO-cond-full-feature" in methods
