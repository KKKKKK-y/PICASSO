"""Stage 2BC runner and result table smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.evaluation import ResultTable  # noqa: E402
from picasso_csi.losses import picasso_generator_loss  # noqa: E402
from picasso_csi.training.run_stage2bc_comprehensive import _select_scope  # noqa: E402


def test_picasso_loss_modes_are_finite() -> None:
    H_true = torch.randn(2, 2, 4, 4, 16)
    H_sparse = H_true * 0.5
    mask = torch.ones_like(H_true)
    H_hat = H_sparse + 0.1 * torch.randn_like(H_sparse)
    fake_logits = torch.randn(2, 1)

    for mode in ["rec_only", "rec_physics", "rec_adv", "full"]:
        losses = picasso_generator_loss(H_hat, H_true, H_sparse, mask, fake_logits=fake_logits, loss_mode=mode)
        assert torch.isfinite(losses["total"])


def test_result_table_writes_paper_rows(tmp_path: Path) -> None:
    table = ResultTable()
    for method, nmse in [("LS", 0.8), ("DnCNN", 0.7), ("PICASSO-rec", 0.75)]:
        table.append(
            {
                "stage": "test",
                "method": method,
                "loss_mode": "",
                "use_condition": False,
                "pilot_ratio": "0.1250",
                "snr_db": "10",
                "seed": 42,
                "nmse": f"{nmse:.8f}",
                "mse": "0.1",
                "mae": "0.1",
                "pilot_consistency_error": "0.1",
                "delay_sparsity_score": "0.1",
                "epochs": 1,
                "num_train_samples": 8,
                "runtime_seconds": "0.1",
            }
        )

    raw = table.save_raw_csv(tmp_path / "raw.csv")
    paper = table.save_paper_table_csv(tmp_path / "paper.csv")

    assert raw.exists()
    assert paper.exists()
    assert "DnCNN" in paper.read_text(encoding="utf-8")


def test_stage2bc_scope_reduces_large_grid() -> None:
    config = {
        "data": {
            "seeds": [1, 2, 3],
            "pilot_ratios": [0.5, 0.25, 0.125, 0.0625],
            "snr_db_values": [10, 20, 30],
            "train_samples": 4096,
            "test_samples": 1024,
        },
        "training": {"epochs_main": 20, "epochs_diagnostic": 8},
        "diagnosis": {
            "allow_reduced_grid_if_runtime_high": True,
            "reduced_grid_seeds": [1],
            "reduced_grid_pilot_ratios": [0.125],
            "reduced_grid_snr_db_values": [10],
            "reduced_train_samples": 8,
            "reduced_test_samples": 4,
            "reduced_epochs_main": 1,
            "reduced_epochs_diagnostic": 1,
        },
        "models": {
            "run_ls": True,
            "run_dncnn": True,
            "run_cond_dncnn": True,
            "run_enhanced_dncnn": True,
            "run_picasso_rec": True,
            "run_picasso_rec_physics": True,
            "run_picasso_rec_adv": True,
            "run_picasso_full": True,
            "run_picasso_cond_full": True,
        },
    }

    scope = _select_scope(config)

    assert scope["name"] == "reduced"
    assert scope["train_samples"] == 8
