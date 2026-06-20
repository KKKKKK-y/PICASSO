import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.training.run_stage5a_ablation import (
    _load_config,
    _validate_config,
    build_ablation_table,
    load_existing_csvs,
    relative_change,
    write_markdown_report,
)


CONFIG = ROOT / "configs" / "stage5a_ablation.yaml"


def test_stage5a_config_is_readable_and_artifact_safe() -> None:
    config = _load_config(CONFIG)
    assert config["project"]["stage"] == "stage5a_paper_level_ablation"
    _validate_config(config)
    assert config["artifact_policy"]["save_checkpoints"] is False


def test_existing_csv_loader_finds_primary_stages() -> None:
    loaded = load_existing_csvs(ROOT / "outputs" / "results")
    assert {"stage2bc", "stage3a_larger", "stage3b", "stage4"}.issubset(loaded)
    assert all(loaded.values())


def test_ablation_table_builder_and_relative_change() -> None:
    categories = {
        "model": [
            {"method": "PICASSO-rec", "nmse": "0.8"},
            {"method": "Enhanced-DnCNN", "nmse": "1.0"},
        ]
    }
    table = build_ablation_table(categories)
    assert len(table) == 2
    assert relative_change(0.8, 1.0) == pytest.approx(-0.2)
    picasso = next(row for row in table if row["Variant"] == "PICASSO-rec")
    assert picasso["Keep in Final Model?"] == "yes"


def test_relative_change_rejects_zero_reference() -> None:
    with pytest.raises(ValueError):
        relative_change(1.0, 0.0)


def test_markdown_report_writer_creates_no_checkpoint(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    categories = {name: [] for name in ["model", "architecture", "loss", "gan", "channel", "robustness"]}
    categories["model"] = [{"method": "PICASSO-rec", "nmse": "0.8"}]
    output = write_markdown_report(report, categories, ["stage4"], 0)
    assert output.exists()
    assert "PICASSO-rec remains the final model" in output.read_text(encoding="utf-8")
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))
    assert not list(tmp_path.rglob("*.ckpt"))


def test_unsafe_config_is_rejected() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["artifact_policy"]["save_checkpoints"] = True
    with pytest.raises(ValueError):
        _validate_config(config)
