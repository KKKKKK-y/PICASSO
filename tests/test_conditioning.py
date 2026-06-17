"""Tests for condition-aware model inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.models import DnCNNBaseline, PICASSOGenerator, make_condition_channels  # noqa: E402


def test_condition_channels_shape_and_finite_values() -> None:
    channels = make_condition_channels(3, 16, 64, torch.tensor([10.0, 20.0, 30.0]), 0.25)

    assert channels.shape == (3, 2, 16, 64)
    assert torch.isfinite(channels).all()
    assert torch.all(channels[:, 0] >= 0.0)
    assert torch.all(channels[:, 0] <= 1.0)
    assert torch.allclose(channels[:, 1], torch.full_like(channels[:, 1], 0.25))


def test_conditioned_dncnn_forward_shape() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    mask = torch.ones_like(H_sparse)
    model = DnCNNBaseline(hidden_channels=8, depth=3, use_condition=True)

    H_hat = model(H_sparse, mask, torch.tensor([10.0, 20.0]), torch.tensor([0.25, 0.125]))

    assert H_hat.shape == H_sparse.shape
    assert torch.isfinite(H_hat).all()


def test_conditioned_picasso_forward_shape() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    model = PICASSOGenerator(base_channels=8, num_blocks=1, use_condition=True)

    H_hat = model(H_sparse, torch.tensor([10.0, 20.0]), torch.tensor([0.25, 0.125]))

    assert H_hat.shape == H_sparse.shape
    assert torch.isfinite(H_hat).all()
