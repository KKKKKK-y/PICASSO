"""Tests for Stage 1B evaluation metrics and artifact policy."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import SyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import (  # noqa: E402
    delay_domain_sparsity_score,
    mae,
    mse,
    nmse,
    pilot_consistency_error,
)


def test_metrics_return_finite_scalars_for_batched_tensors() -> None:
    dataset = SyntheticCSIDataset(num_samples=1, seed=42)
    sample = dataset[0]
    H_sparse = sample["H_sparse"].unsqueeze(0)
    H_full = sample["H_full"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    values = [
        mse(H_sparse, H_full),
        mae(H_sparse, H_full),
        nmse(H_sparse, H_full),
        pilot_consistency_error(H_sparse, H_sparse, mask),
        delay_domain_sparsity_score(H_full),
    ]

    for value in values:
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_metrics_return_finite_scalars_for_non_batched_tensors() -> None:
    dataset = SyntheticCSIDataset(num_samples=1, seed=123)
    sample = dataset[0]

    values = [
        nmse(sample["H_sparse"], sample["H_full"]),
        delay_domain_sparsity_score(sample["H_full"]),
    ]

    for value in values:
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_no_protected_artifacts_are_present() -> None:
    protected_extensions = {
        ".pt",
        ".pth",
        ".ckpt",
        ".npy",
        ".npz",
        ".mat",
        ".h5",
        ".zip",
        ".tar",
        ".gz",
    }
    protected_files = [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and (path.suffix in protected_extensions or path.name.endswith(".tar.gz"))
        and ".git" not in path.parts
    ]

    assert protected_files == []
