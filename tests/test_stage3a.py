"""Tests for Stage 3A supervised physics-guided reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.evaluation import STAGE3A_FIELDS, ResultTable  # noqa: E402
from picasso_csi.losses import picasso_generator_loss  # noqa: E402
from picasso_csi.training.run_stage3a_supervised_physics import _select_scope  # noqa: E402


def test_stage3a_config_is_readable() -> None:
    config_path = PROJECT_ROOT / "configs" / "stage3a_supervised_physics.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["project"]["stage"] == "stage3a_supervised_physics"
    assert config["artifact_policy"]["save_checkpoints"] is False
    assert config["artifact_policy"]["save_numpy_arrays"] is False


def test_stage3a_picasso_loss_modes_are_finite() -> None:
    H_true = torch.randn(2, 2, 4, 4, 16)
    H_sparse = H_true * 0.5
    mask = torch.ones_like(H_true)
    H_hat = H_sparse + 0.05 * torch.randn_like(H_sparse)
    fake_logits = torch.randn(2, 1)

    rec = picasso_generator_loss(H_hat, H_true, H_sparse, mask, loss_mode="rec_only")
    physics = picasso_generator_loss(H_hat, H_true, H_sparse, mask, loss_mode="rec_physics")
    light_adv = picasso_generator_loss(
        H_hat,
        H_true,
        H_sparse,
        mask,
        fake_logits=fake_logits,
        loss_mode="full",
        lambda_adv=0.001,
    )

    assert torch.isfinite(rec["total"])
    assert torch.isfinite(physics["total"])
    assert torch.isfinite(light_adv["total"])


def test_stage3a_result_table_writer(tmp_path: Path) -> None:
    table = ResultTable()
    table.append(
        {
            "stage": "stage3a_supervised_physics",
            "seed": 42,
            "method": "PICASSO-rec-physics",
            "pilot_ratio": "0.1250",
            "snr_db": "10",
            "nmse": "0.80000000",
            "mse": "0.10000000",
            "mae": "0.20000000",
            "pilot_consistency_error": "0.01000000",
            "delay_sparsity_score": "0.90000000",
            "epochs_used": 1,
            "train_samples": 8,
            "val_samples": 4,
            "test_samples": 4,
            "runtime_seconds": "0.1",
            "loss_mode": "rec_physics",
            "lambda_adv": 0.0,
            "lambda_pilot": 1.0,
            "lambda_smooth": 0.05,
            "lambda_sparse": 0.05,
        }
    )

    raw = table.save_raw_csv_with_fields(tmp_path / "stage3a_raw.csv", STAGE3A_FIELDS)
    summary = table.save_summary_by_condition_csv(tmp_path / "stage3a_summary.csv")
    paper = table.save_paper_table_csv(tmp_path / "stage3a_paper.csv")

    assert raw.exists()
    assert summary.exists()
    assert paper.exists()
    assert "PICASSO-rec-physics" in paper.read_text(encoding="utf-8")


def test_stage3a_scope_reduces_large_budget() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "stage3a_supervised_physics.yaml").read_text(encoding="utf-8"))

    scope = _select_scope(config)

    assert scope["name"] == "reduced"
    assert scope["train_samples"] <= config["data"]["train_samples"]


def test_no_protected_stage3a_artifacts_exist() -> None:
    protected_extensions = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".mat", ".h5", ".zip", ".tar", ".gz"}
    protected_files = [
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and (path.suffix in protected_extensions or path.name.endswith(".tar.gz"))
        and ".git" not in path.parts
    ]

    assert protected_files == []
