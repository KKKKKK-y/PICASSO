"""Tests for Stage 2BC random channel difficulty controls."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import NoisySyntheticCSIDataset  # noqa: E402


def _dataset(seed: int, snr_db: float) -> NoisySyntheticCSIDataset:
    return NoisySyntheticCSIDataset(
        num_samples=3,
        pilot_ratio=0.25,
        snr_db=snr_db,
        seed=seed,
        random_n_paths=True,
        min_paths=3,
        max_paths=10,
        delay_spread="random",
        normalize_channel=True,
        pilot_noise_only=True,
    )


def test_random_path_count_dataset_runs() -> None:
    sample = _dataset(seed=11, snr_db=20)[0]

    assert sample["H_full"].shape == (2, 4, 4, 64)
    assert sample["H_sparse"].shape == (2, 4, 4, 64)
    assert torch.isfinite(sample["H_full"]).all()


def test_seed_reproducibility_and_difference() -> None:
    dataset_a = _dataset(seed=12, snr_db=20)
    dataset_b = _dataset(seed=12, snr_db=20)
    dataset_c = _dataset(seed=13, snr_db=20)

    assert torch.allclose(dataset_a[1]["H_full"], dataset_b[1]["H_full"])
    assert not torch.allclose(dataset_a[1]["H_full"], dataset_c[1]["H_full"])


def test_lower_snr_has_larger_pilot_noise() -> None:
    low = _dataset(seed=14, snr_db=10)[0]
    high = _dataset(seed=14, snr_db=30)[0]

    low_delta = torch.mean(torch.abs((low["H_sparse"] - low["H_full"]) * low["mask"]))
    high_delta = torch.mean(torch.abs((high["H_sparse"] - high["H_full"]) * high["mask"]))

    assert low_delta > high_delta


def test_dataset_creation_does_not_save_files() -> None:
    before = {path for path in PROJECT_ROOT.rglob("*") if path.is_file()}
    _ = _dataset(seed=15, snr_db=20)
    after = {path for path in PROJECT_ROOT.rglob("*") if path.is_file()}

    assert after == before
