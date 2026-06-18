"""Tests for Stage 4 CDL-inspired channel generalization."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import CDLChannelDataset  # noqa: E402
from picasso_csi.evaluation import doppler_robustness_metric  # noqa: E402
from picasso_csi.simulation import create_pilot_mask, generate_cdl_mimo_ofdm_channel  # noqa: E402
from picasso_csi.training.run_stage4 import _scope  # noqa: E402


def test_cdl_channel_shape_and_profiles() -> None:
    for profile in ["CDL-A", "CDL-B", "CDL-C"]:
        channel = generate_cdl_mimo_ofdm_channel(4, 4, 64, profile=profile, velocity_kmh=30, seed=7)
        assert channel.shape == (4, 4, 64, 2)
        assert torch.isfinite(torch.from_numpy(channel)).all()


def test_cdl_dataset_fields_and_shapes() -> None:
    dataset = CDLChannelDataset(
        num_samples=3,
        profile="CDL-B",
        pilot_ratio=0.125,
        pilot_pattern="irregular",
        snr_db=10,
        velocity_kmh=60,
        seed=9,
    )
    sample = dataset[0]

    assert {"H_full", "H_prev", "H_sparse", "mask", "snr_db", "pilot_ratio", "velocity_kmh", "profile_id"} == set(sample)
    assert sample["H_full"].shape == (2, 4, 4, 64)
    assert sample["H_prev"].shape == (2, 4, 4, 64)
    assert sample["H_sparse"].shape == (2, 4, 4, 64)


def test_pilot_patterns_are_supported() -> None:
    for pattern in ["comb", "block", "irregular"]:
        mask = create_pilot_mask(4, 4, 64, 0.25, pattern=pattern)
        assert mask.shape == (4, 4, 64, 2)
        assert mask.sum() > 0


def test_doppler_metric_is_finite() -> None:
    H_prev = torch.randn(2, 2, 4, 4, 16)
    H_full = H_prev + 0.1 * torch.randn_like(H_prev)
    H_hat = H_full + 0.05 * torch.randn_like(H_full)

    value = doppler_robustness_metric(H_hat, H_full, H_prev)

    assert torch.isfinite(value)


def test_doppler_metric_handles_static_channels() -> None:
    H_full = torch.randn(2, 2, 4, 4, 16)
    H_hat = H_full + 0.05 * torch.randn_like(H_full)

    value = doppler_robustness_metric(H_hat, H_full, H_full)

    assert torch.isfinite(value)
    assert value < 1.0


def test_stage4_config_reduces_large_grid() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "stage4_cdl.yaml").read_text(encoding="utf-8"))
    scope = _scope(config)

    assert config["project"]["stage"] == "stage4_cdl"
    assert scope["name"] == "reduced"
    assert set(scope["profiles"]) == {"CDL-A", "CDL-B", "CDL-C"}
