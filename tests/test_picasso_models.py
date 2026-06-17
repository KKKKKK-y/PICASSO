"""Tests for PICASSO generator and discriminator skeletons."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.models import PICASSODiscriminator, PICASSOGenerator  # noqa: E402


def test_picasso_generator_and_discriminator_shapes() -> None:
    H_sparse = torch.randn(4, 2, 4, 4, 64)
    generator = PICASSOGenerator(base_channels=8, num_blocks=1)
    discriminator = PICASSODiscriminator(base_channels=8)

    H_hat = generator(H_sparse)
    logits = discriminator(H_hat)

    assert H_hat.shape == H_sparse.shape
    assert logits.shape == (4, 1)
    assert torch.isfinite(H_hat).all()
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_picasso_models_support_cuda() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64, device="cuda")
    generator = PICASSOGenerator(base_channels=8, num_blocks=1).cuda()
    discriminator = PICASSODiscriminator(base_channels=8).cuda()

    H_hat = generator(H_sparse)
    logits = discriminator(H_hat)

    assert H_hat.is_cuda
    assert logits.is_cuda
    assert H_hat.shape == H_sparse.shape
    assert logits.shape == (2, 1)
    assert torch.isfinite(H_hat).all()
    assert torch.isfinite(logits).all()
