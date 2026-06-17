"""Tests for Stage 1A synthetic CSI datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import SyntheticCSIDataset  # noqa: E402


def test_synthetic_csi_dataset_sample_and_batch_shapes() -> None:
    dataset = SyntheticCSIDataset(num_samples=8, seed=123)
    sample = dataset[0]

    assert set(sample) == {"H_sparse", "H_full", "mask"}
    assert sample["H_sparse"].shape == (2, 4, 4, 64)
    assert sample["H_full"].shape == (2, 4, 4, 64)
    assert sample["mask"].shape == (2, 4, 4, 64)
    assert sample["H_sparse"].dtype == torch.float32
    assert sample["H_full"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["H_sparse"].shape == (4, 2, 4, 4, 64)
    assert batch["H_full"].shape == (4, 2, 4, 4, 64)
    assert batch["mask"].shape == (4, 2, 4, 4, 64)


def test_synthetic_csi_dataset_is_reproducible() -> None:
    dataset_a = SyntheticCSIDataset(num_samples=2, seed=7)
    dataset_b = SyntheticCSIDataset(num_samples=2, seed=7)

    assert torch.allclose(dataset_a[1]["H_full"], dataset_b[1]["H_full"])
    assert torch.allclose(dataset_a[1]["H_sparse"], dataset_b[1]["H_sparse"])
