"""Tests for Stage 1A/1B baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import SyntheticCSIDataset  # noqa: E402
from picasso_csi.models import (  # noqa: E402
    CNNBaseline,
    DnCNNBaseline,
    LSInterpolationBaseline,
    lmmse_like_baseline,
    ls_baseline,
    omp_like_baseline,
)


def test_stage1a_module_baseline_shapes() -> None:
    dataset = SyntheticCSIDataset(num_samples=2, seed=42)
    sample = dataset[0]
    H_sparse = sample["H_sparse"].unsqueeze(0)
    H_full = sample["H_full"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    baselines = [
        LSInterpolationBaseline(),
        CNNBaseline(hidden_channels=8),
        DnCNNBaseline(hidden_channels=8),
    ]

    for baseline in baselines:
        H_hat = baseline(H_sparse, mask)
        assert H_hat.shape == H_full.shape



def test_stage1b_classical_baseline_shapes() -> None:
    dataset = SyntheticCSIDataset(num_samples=2, seed=42)
    sample = dataset[0]
    H_sparse = sample["H_sparse"].unsqueeze(0)
    H_full = sample["H_full"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    estimates = [
        ls_baseline(H_sparse, mask),
        lmmse_like_baseline(H_sparse, mask),
        omp_like_baseline(H_sparse, mask, n_paths=6),
    ]

    for H_hat in estimates:
        assert H_hat.shape == H_full.shape
        assert torch.isfinite(H_hat).all()


def test_stage1b_non_batched_baseline_shapes() -> None:
    dataset = SyntheticCSIDataset(num_samples=1, seed=13)
    sample = dataset[0]
    H_sparse = sample["H_sparse"]
    mask = sample["mask"]

    assert ls_baseline(H_sparse, mask).shape == H_sparse.shape
    assert lmmse_like_baseline(H_sparse, mask).shape == H_sparse.shape
    assert omp_like_baseline(H_sparse, mask, n_paths=6).shape == H_sparse.shape
