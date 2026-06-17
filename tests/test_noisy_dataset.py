"""Tests for Stage 2A noise-aware synthetic CSI dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import NoisySyntheticCSIDataset  # noqa: E402


def test_noisy_dataset_fields_and_shapes() -> None:
    dataset = NoisySyntheticCSIDataset(num_samples=4, pilot_ratio=0.25, snr_db=20, seed=123)
    sample = dataset[0]

    assert set(sample) == {"H_full", "H_sparse", "mask", "snr_db", "pilot_ratio"}
    assert sample["H_full"].shape == (2, 4, 4, 64)
    assert sample["H_sparse"].shape == (2, 4, 4, 64)
    assert sample["mask"].shape == (2, 4, 4, 64)
    assert sample["snr_db"].item() == 20.0
    assert sample["pilot_ratio"].item() == 0.25
    assert torch.isfinite(sample["H_full"]).all()
    assert torch.isfinite(sample["H_sparse"]).all()


def test_noisy_dataset_adds_finite_noise_only_on_pilots() -> None:
    dataset = NoisySyntheticCSIDataset(num_samples=2, pilot_ratio=0.25, snr_db=10, seed=7)
    sample = dataset[0]
    H_full = sample["H_full"]
    H_sparse = sample["H_sparse"]
    mask = sample["mask"]

    assert torch.allclose(H_sparse[mask == 0.0], torch.zeros_like(H_sparse[mask == 0.0]))
    pilot_delta = torch.abs(H_sparse[mask == 1.0] - H_full[mask == 1.0])
    assert torch.isfinite(pilot_delta).all()
    assert torch.mean(pilot_delta) > 0.0


def test_noisy_dataset_is_reproducible_for_same_seed() -> None:
    dataset_a = NoisySyntheticCSIDataset(num_samples=2, pilot_ratio=0.125, snr_db=30, seed=42)
    dataset_b = NoisySyntheticCSIDataset(num_samples=2, pilot_ratio=0.125, snr_db=30, seed=42)

    assert torch.allclose(dataset_a[1]["H_full"], dataset_b[1]["H_full"])
    assert torch.allclose(dataset_a[1]["H_sparse"], dataset_b[1]["H_sparse"])
    assert torch.allclose(dataset_a[1]["mask"], dataset_b[1]["mask"])


def test_no_protected_artifacts_are_present_after_dataset_creation() -> None:
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
