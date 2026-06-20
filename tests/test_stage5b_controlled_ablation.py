"""Tests for the strict Stage 5B controlled ablation runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.training.run_stage5b_controlled_ablation import (  # noqa: E402
    _load_config,
    _validate_config,
    add_deltas_and_labels,
    build_final_table,
    build_variants,
    contribution_label,
    write_report,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "stage5b_controlled_ablation.yaml"


def test_config_is_readable_and_strict() -> None:
    config = _load_config(CONFIG_PATH)
    _validate_config(config)
    assert config["experiment"]["controlled"] is True
    assert config["training"]["epochs"] == 30
    assert config["training"]["same_training_budget_for_all_variants"] is True


def test_variant_builder_has_requested_unique_matrix() -> None:
    variants = build_variants()
    assert len(variants) == 20
    assert len({variant["key"] for variant in variants}) == 20
    group_counts = {
        group: sum(group in variant["groups"] for variant in variants)
        for group in ["baseline", "architecture", "loss", "gan"]
    }
    assert group_counts == {"baseline": 3, "architecture": 8, "loss": 8, "gan": 4}


def test_architecture_variants_change_only_declared_flags() -> None:
    variants = {variant["key"]: variant for variant in build_variants()}
    flags = {"refinement", "multiscale", "attention", "film"}
    assert flags.intersection(variants["base"]) == set()
    assert flags.intersection(variants["refinement"]) == {"refinement"}
    assert flags.intersection(variants["multiscale"]) == {"multiscale"}
    assert flags.intersection(variants["se"]) == {"attention"}
    assert flags.intersection(variants["film"]) == {"film"}
    assert flags.intersection(variants["refinement_multiscale"]) == {"refinement", "multiscale"}
    assert flags.intersection(variants["refinement_multiscale_se"]) == {"refinement", "multiscale", "attention"}
    assert flags.intersection(variants["enhanced_full"]) == flags


def test_loss_and_gan_variants_do_not_enable_architecture_flags() -> None:
    for variant in build_variants():
        if variant["key"].startswith("loss_") or variant["key"].startswith("gan_"):
            assert not any(variant.get(flag, False) for flag in ["refinement", "multiscale", "attention", "film"])
        if variant["key"].startswith("loss_"):
            assert "gan" not in variant
        if variant["key"].startswith("gan_"):
            assert "loss_components" not in variant


def test_delta_and_contribution_labels_are_correct() -> None:
    rows = [
        _row("baseline", "PICASSO-rec-base", 0.8),
        _row("architecture", "variant-positive", 0.798),
        _row("architecture", "variant-neutral", 0.8005),
        _row("architecture", "variant-negative", 0.802),
    ]
    output = add_deltas_and_labels(rows)
    labels = {row["variant"]: row["contribution_label"] for row in output}
    assert labels["variant-positive"] == "positive"
    assert labels["variant-neutral"] == "neutral"
    assert labels["variant-negative"] == "negative"
    assert contribution_label(-0.0011) == "positive"
    assert contribution_label(0.001) == "neutral"
    assert contribution_label(0.0011) == "negative"


def test_final_table_and_markdown_writer(tmp_path: Path) -> None:
    rows = _complete_rows()
    final = build_final_table(rows)
    assert 7 <= len(final) <= 8
    assert {row["Group"] for row in final} == {"baseline", "architecture", "loss", "gan"}
    report = tmp_path / "stage5b.md"
    output = write_report(report, rows, _load_config(CONFIG_PATH), 12.0)
    text = output.read_text(encoding="utf-8")
    assert "strict controlled ablation" in text
    assert "PICASSO-rec remains" in text
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.ckpt"))


def test_unsafe_artifact_policy_is_rejected() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["artifact_policy"]["save_predictions"] = True
    with pytest.raises(ValueError):
        _validate_config(config)


def _row(group: str, variant: str, nmse: float) -> dict[str, object]:
    return {
        "ablation_group": group, "variant": variant, "channel_profile": "CDL-A", "seed": 42,
        "pilot_ratio": "0.1250", "snr_db": "10", "velocity": "60", "pilot_pattern": "comb",
        "nmse": f"{nmse:.8f}", "mse": "0.4", "mae": "0.5", "pilot_consistency_error": "0.1",
        "doppler_robustness": "0.9", "runtime_seconds": "1.0", "num_parameters": 10,
        "delta_nmse_vs_base": f"{nmse - 0.8:.8f}", "contribution_label": contribution_label(nmse - 0.8),
    }


def _complete_rows() -> list[dict[str, object]]:
    rows = []
    offset = 0
    for variant in build_variants():
        for group, alias in variant["groups"].items():
            value = 0.8 + offset * 0.0001
            rows.append(_row(group, alias, value))
        offset += 1
    return rows
