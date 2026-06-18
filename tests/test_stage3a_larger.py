"""Tests for Stage 3A-L larger PICASSO training utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.evaluation import write_csv  # noqa: E402
from picasso_csi.models import DnCNNBaseline, EnhancedDnCNNBaseline, PICASSODiscriminator, PICASSOGenerator  # noqa: E402
from picasso_csi.training.run_stage3a_larger_training import CURVE_FIELDS, _count_parameters, _select_scope  # noqa: E402


def test_larger_generator_forward_shape() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    model = PICASSOGenerator(base_channels=64, num_blocks=6)

    H_hat = model(H_sparse)

    assert H_hat.shape == H_sparse.shape
    assert torch.isfinite(H_hat).all()


def test_larger_conditional_generator_forward_shape() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    model = PICASSOGenerator(base_channels=64, num_blocks=6, use_condition=True)

    H_hat = model(H_sparse, torch.tensor([10.0, 20.0]), torch.tensor([0.125, 0.0625]))

    assert H_hat.shape == H_sparse.shape
    assert torch.isfinite(H_hat).all()


def test_larger_discriminator_forward_shape() -> None:
    H = torch.randn(2, 2, 4, 4, 64)
    model = PICASSODiscriminator(base_channels=64)

    logits = model(H)

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_enhanced_dncnn_forward_shape_and_param_count() -> None:
    H_sparse = torch.randn(2, 2, 4, 4, 64)
    mask = torch.ones_like(H_sparse)
    base = DnCNNBaseline()
    enhanced = EnhancedDnCNNBaseline(hidden_channels=64, depth=8)

    H_hat = enhanced(H_sparse, mask)

    assert H_hat.shape == H_sparse.shape
    assert _count_parameters(enhanced) > _count_parameters(base)


def test_stage3a_larger_config_and_scope() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "stage3a_larger_training.yaml").read_text(encoding="utf-8"))
    scope = _select_scope(config)

    assert config["project"]["stage"] == "stage3a_larger_training"
    assert config["checkpoint"]["commit_checkpoints"] is False
    assert scope["name"] == "reduced"


def test_training_curve_writer(tmp_path: Path) -> None:
    path = write_csv(
        [
            {
                "stage": "stage3a_larger_training",
                "seed": 42,
                "method": "PICASSO-full",
                "pilot_ratio": "0.1250",
                "snr_db": "10",
                "epoch": 1,
                "train_loss": "0.1",
                "val_nmse": "0.8",
                "learning_rate": "0.0001",
            }
        ],
        tmp_path / "curves.csv",
        CURVE_FIELDS,
    )

    assert path.exists()
    assert "val_nmse" in path.read_text(encoding="utf-8")


def test_checkpoint_dir_is_gitignored() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "checkpoints/" in gitignore
    assert "*.pt" in gitignore
