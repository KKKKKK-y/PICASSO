"""Tests for Stage 1A baselines and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import SyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import mae, mse, nmse, pilot_consistency_error  # noqa: E402
from picasso_csi.models import CNNBaseline, DnCNNBaseline, LSInterpolationBaseline  # noqa: E402


def test_stage1a_baseline_shapes_and_metrics() -> None:
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

    H_hat = baselines[1](H_sparse, mask)
    metric_values = [
        mse(H_hat, H_full),
        mae(H_hat, H_full),
        nmse(H_hat, H_full),
        pilot_consistency_error(H_hat, H_sparse, mask),
    ]
    for value in metric_values:
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert torch.isfinite(value)
