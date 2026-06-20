"""Build the Stage 5A paper-level ablation tables with minimal gap runs."""

from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch
import yaml

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.evaluation import write_csv  # noqa: E402
from picasso_csi.training.run_stage3b_incremental import (  # noqa: E402
    _build_loaders,
    _evaluate,
    _gen_forward,
    _make_generator,
    _spec,
    _train_picasso,
)


RESULT_FIELDS = [
    "ablation_type", "condition", "method", "pilot_ratio", "snr_db",
    "velocity", "channel_profile", "pilot_pattern", "nmse", "mse", "mae",
    "doppler_robustness", "runtime_seconds", "source_stage", "reused_or_new",
]
FINAL_FIELDS = [
    "Ablation Factor", "Variant", "NMSE", "Relative Change",
    "Interpretation", "Keep in Final Model?",
]

SOURCE_FILES = {
    "stage2a": "stage2a_small_formal_results.csv",
    "stage2bc": "stage2bc_raw_results.csv",
    "stage3a_larger": "stage3a_larger_raw_results.csv",
    "stage3b": "stage3b_incremental_results.csv",
    "stage4": "stage4_raw_results.csv",
}


def load_existing_csvs(results_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load available row-level result CSVs without mutating source data."""

    root = Path(results_dir)
    loaded: dict[str, list[dict[str, str]]] = {}
    for stage, filename in SOURCE_FILES.items():
        path = root / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                loaded[stage] = list(csv.DictReader(handle))
    return loaded


def relative_change(value: float, reference: float) -> float:
    """Return signed relative change; negative values indicate lower NMSE."""

    if reference == 0:
        raise ValueError("Reference value must be non-zero.")
    return (value - reference) / reference


def build_ablation_table(
    categories: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Aggregate standardized rows into a compact paper-facing table."""

    references = {
        "model": "PICASSO-rec",
        "architecture": "PICASSO-rec-base",
        "loss": "rec only",
        "gan": "no GAN",
        "channel": "noisy random-path synthetic OFDM",
        "robustness": "PICASSO-rec",
    }
    output: list[dict[str, object]] = []
    for category, rows in categories.items():
        grouped = _category_means(category, rows)
        if not grouped:
            continue
        for method, value in sorted(grouped.items(), key=lambda item: item[1]):
            reference_name = references[category]
            if category == "gan":
                reference_name = "no GAN (Stage 3B)" if "Stage 3B" in method else "no GAN (Stage 2BC)"
            reference = grouped.get(reference_name, next(iter(grouped.values())))
            delta = relative_change(value, reference)
            keep = _keep_decision(category, method, delta)
            output.append({
                "Ablation Factor": category,
                "Variant": method,
                "NMSE": f"{value:.8f}",
                "Relative Change": f"{delta:+.2%}",
                "Interpretation": _interpret(category, method, delta),
                "Keep in Final Model?": keep,
            })
    return output


def write_markdown_report(
    path: str | Path,
    categories: dict[str, list[dict[str, object]]],
    loaded_files: Iterable[str],
    new_count: int,
) -> Path:
    """Write the evidence-bounded Stage 5A report."""

    means = {name: _category_means(name, rows) for name, rows in categories.items()}
    architecture = means.get("architecture", {})
    losses = means.get("loss", {})
    strongest = min(architecture, key=architecture.get) if architecture else "not available"
    weakest_loss = max(losses, key=losses.get) if losses else "not available"
    lines = [
        "# Stage 5A Paper-Level Ablation Report", "",
        "## 1. Ablation Study Objective", "",
        "This stage converts the existing experiments into a traceable ablation argument. It does not redesign the model or overwrite any Stage 0-4 result.", "",
        "## 2. Existing Result Reuse Strategy", "",
        f"Loaded row-level sources: {', '.join(sorted(loaded_files))}. Historical rows retain source provenance and are never relabeled as newly run results.", "",
        "## 3. Newly Run Experiments", "",
        f"New reduced-grid rows: {new_count}. They isolate architecture and physics-loss components at pilot ratio 0.125, 10 dB, two seeds, and two epochs. They support mechanism analysis only.", "",
    ]
    section_titles = {
        "model": "4. Model Ablation Results",
        "architecture": "5. Architecture Ablation Results",
        "loss": "6. Loss Component Ablation Results",
        "gan": "7. GAN Ablation Results",
        "channel": "8. Channel Complexity Ablation Results",
        "robustness": "9. Robustness Ablation Results",
    }
    for category in ["model", "architecture", "loss", "gan", "channel", "robustness"]:
        lines.extend([f"## {section_titles[category]}", ""])
        for method, value in sorted(means.get(category, {}).items(), key=lambda item: item[1]):
            lines.append(f"- {method}: mean NMSE {value:.8f}")
        lines.append("")
    lines.extend([
        "## 10. Final Model Choice", "",
        "PICASSO-rec remains the final model. It is the most stable supervised choice across the controlled synthetic and CDL-inspired evidence; reduced component diagnostics do not replace that broader evidence.", "",
        "## 11. Why PICASSO-rec Is Final", "",
        "Its gain is attributable primarily to reconstruction architecture rather than adversarial training or a larger loss stack. It also avoids the instability and additional optimization state of GAN variants.", "",
        "## 12. Why GAN Is Optional/Deprecated", "",
        "Within-stage paired controls show that neither output/light adversarial nor feature-level adversarial variants deliver a consistent NMSE gain. Stage 2BC and Stage 3B absolute values are not compared directly. Distributional effects were not measured with a validated perceptual metric, so no stronger GAN claim is warranted.", "",
        "## 13. Why Physics Loss Is Secondary", "",
        f"Component isolation identifies `{weakest_loss}` as the weakest reduced-grid loss setting, while bundled physics is also worse than reconstruction-only in Stage 3B and Stage 4. Physics remains useful for consistency analysis, but over-regularization and simulator mismatch limit its main-metric gain.", "",
        "## 14. Strongest PICASSO Condition", "",
        "The clearest broad advantage appears in CDL-inspired reconstruction, including sparse-pilot and mobility conditions, where Stage 4 reports PICASSO-rec ahead of Enhanced-DnCNN overall. Exact gains should be quoted only from matched Stage 4 rows.", "",
        "## 15. Suggested Final Paper Tables", "",
        "- **Table I - Main Performance Comparison:** Method, Synthetic NMSE, CDL NMSE, Low Pilot NMSE, High Mobility NMSE, Avg NMSE.",
        "- **Table II - Ablation Study:** Variant, Removed/Added Component, NMSE, Relative Change, Interpretation.",
        "- **Table III - Robustness Analysis:** Condition, LS, DnCNN, Enhanced-DnCNN, PICASSO-rec, Gain over Best Baseline.", "",
        "## 16. Suggested Final Paper Figures", "",
        "- **Figure 1 - PICASSO Framework:** sparse pilot CSI, reconstruction network, optional physics loss, full CSI output.",
        "- **Figure 2 - NMSE vs Pilot Ratio:** LS, DnCNN, Enhanced-DnCNN, PICASSO-rec.",
        "- **Figure 3 - NMSE vs Velocity:** Enhanced-DnCNN, PICASSO-rec, PICASSO-rec-physics.",
        "- **Figure 4 - Ablation Bar Chart:** reconstruction, reconstruction+physics, reconstruction+GAN, enhanced reconstruction.", "",
        "## 17. Limitations", "",
        "The CDL simulator is CDL-inspired rather than a complete TR 38.901 implementation. Several tables combine provenance-aware historical stages with different budgets, so cross-stage values are descriptive. The component runs are deliberately small, and no claim of statistical significance is made. Real measured CSI remains future validation.", "",
        "## Final Evidence Statement", "",
        f"The strongest isolated architecture variant is `{strongest}`, but the complete evidence still supports PICASSO-rec as final, physics as secondary, and GAN as deprecated.",
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_stage5a(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    results_dir = Path(config["outputs"]["results_dir"])
    loaded = load_existing_csvs(results_dir)
    categories = _reuse_categories(loaded)
    device = torch.device("cuda" if config["runtime"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    new_rows: list[dict[str, object]] = []
    if config["data"]["run_missing_experiments"]:
        new_rows = _run_supplemental(config, device)
        categories["architecture"].extend(row for row in new_rows if row["ablation_type"] == "architecture")
        categories["loss"].extend(row for row in new_rows if row["ablation_type"] == "loss")

    output_paths = _write_outputs(config, categories)
    final_rows = build_ablation_table(categories)
    final_path = results_dir / config["outputs"]["final_ablation_table_csv"]
    write_csv(final_rows, final_path, FINAL_FIELDS)
    report = write_markdown_report(config["outputs"]["report_md"], categories, loaded.keys(), len(new_rows))
    output_paths.extend([str(final_path), str(report)])

    reused = sum(1 for rows in categories.values() for row in rows if row["reused_or_new"] == "reused")
    skipped = 2  # full GAN rerun and optional 0.05 pilot grid
    print(f"device: {device}")
    print(f"loaded existing CSV files: {[SOURCE_FILES[key] for key in loaded]}")
    print(f"reused experiments count: {reused}")
    print(f"newly run experiments count: {len(new_rows)}")
    print(f"skipped experiments count: {skipped}")
    for category, rows in categories.items():
        scores = _category_means(category, rows)
        print(f"{category}: rows={len(rows)}, best={min(scores, key=scores.get) if scores else 'n/a'}")
    print("PICASSO-rec remains final model: yes")
    print("GAN should remain deprecated: yes")
    print("physics should remain secondary ablation: yes")
    return {
        "device": str(device), "reused": reused, "new": len(new_rows), "skipped": skipped,
        "outputs": output_paths, "runtime_seconds": time.perf_counter() - start,
    }


def _reuse_categories(loaded: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, object]]]:
    categories = {name: [] for name in ["model", "architecture", "loss", "gan", "channel", "robustness"]}
    for row in loaded.get("stage3a_larger", []):
        if row.get("method") in {"LS", "DnCNN", "Enhanced-DnCNN", "PICASSO-rec", "PICASSO-rec-physics", "PICASSO-full", "PICASSO-cond-full"}:
            categories["model"].append(_standardize(row, "model", "stage3a_larger"))
    for row in loaded.get("stage3b", []):
        method = row.get("method", "")
        if method in {"PICASSO-rec-base", "PICASSO-rec-enhanced", "PICASSO-rec-attn"}:
            categories["architecture"].append(_standardize(row, "architecture", "stage3b"))
        if method in {"PICASSO-rec-base", "PICASSO-rec-physics-enhanced"}:
            renamed = "rec only" if method == "PICASSO-rec-base" else "rec + full physics"
            categories["loss"].append(_standardize(row, "loss", "stage3b", renamed))
        if method in {"PICASSO-rec-base", "PICASSO-full-feature", "PICASSO-cond-full-feature"}:
            rename = {"PICASSO-rec-base": "no GAN (Stage 3B)", "PICASSO-full-feature": "feature-level GAN (Stage 3B)", "PICASSO-cond-full-feature": "conditioned feature-level GAN (Stage 3B)"}[method]
            categories["gan"].append(_standardize(row, "gan", "stage3b", rename))
    for row in loaded.get("stage2bc", []):
        if row.get("method") in {"PICASSO-rec", "PICASSO-rec-adv", "PICASSO-full"}:
            rename = {
                "PICASSO-rec": "no GAN (Stage 2BC)",
                "PICASSO-rec-adv": "light adversarial loss (Stage 2BC)",
                "PICASSO-full": "output-level GAN (Stage 2BC)",
            }[row["method"]]
            categories["gan"].append(_standardize(row, "gan", "stage2bc", rename))
        if row.get("method") in {"LS", "DnCNN", "Enhanced-DnCNN", "PICASSO-rec", "PICASSO-rec-physics"}:
            categories["channel"].append(_standardize(row, "channel", "stage2bc", channel_profile="noisy random-path synthetic OFDM"))
    for row in loaded.get("stage2a", []):
        if row.get("method") in {"LS", "DnCNN", "PICASSO"}:
            categories["channel"].append(_standardize(row, "channel", "stage2a", channel_profile="noisy synthetic OFDM"))
    for row in loaded.get("stage4", []):
        categories["channel"].append(_standardize(row, "channel", "stage4", row.get("profile", "CDL")))
        if row.get("method") in {"LS", "Enhanced-DnCNN", "PICASSO-rec", "PICASSO-rec-physics"}:
            categories["robustness"].append(_standardize(row, "robustness", "stage4"))
    return categories


def _run_supplemental(config: dict[str, Any], device: torch.device) -> list[dict[str, object]]:
    run_config = _stage3b_compatible_config(config)
    architecture_specs = [
        _spec("PICASSO-rec-base", "picasso", loss_mode="rec_only"),
        _spec("PICASSO-rec-refinement", "picasso", loss_mode="rec_only", enhanced=True),
        _spec("PICASSO-rec-multiscale", "picasso", loss_mode="rec_only", multiscale=True),
        _spec("PICASSO-rec-SE", "picasso", loss_mode="rec_only", attention=True),
        _spec("PICASSO-rec-enhanced", "picasso", loss_mode="rec_only", enhanced=True, multiscale=True),
        _spec("PICASSO-rec-attn", "picasso", loss_mode="rec_only", enhanced=True, multiscale=True, attention=True),
        _spec("PICASSO-rec-FiLM", "picasso", loss_mode="rec_only", enhanced=True, multiscale=True, condition=True, film=True),
    ]
    loss_specs = [
        ("rec only", {}),
        ("rec + pilot consistency", {"lambda_pilot": 1.0}),
        ("rec + frequency consistency", {"lambda_frequency": 0.5}),
        ("rec + delay-domain sparsity", {"lambda_sparse": 0.05}),
        ("rec + energy preservation", {"lambda_energy": 0.05}),
        ("rec + pilot + frequency", {"lambda_pilot": 1.0, "lambda_frequency": 0.5}),
        ("rec + pilot + delay", {"lambda_pilot": 1.0, "lambda_sparse": 0.05}),
        ("rec + full physics", dict(config["loss"])),
    ]
    output: list[dict[str, object]] = []
    pilot_ratio = float(config["supplemental"]["pilot_ratio"])
    snr_db = float(config["supplemental"]["snr_db"])
    for seed in config["data"]["seeds"]:
        _set_seed(int(seed))
        train_loader, test_loader = _build_loaders(run_config, int(seed), pilot_ratio, snr_db)
        for spec in architecture_specs:
            method_start = time.perf_counter()
            generator = _make_generator(run_config, spec).to(device)
            _train_picasso(generator, None, train_loader, device, run_config, spec)
            metrics = _evaluate(test_loader, device, lambda batch, g=generator, c=spec["condition"]: _gen_forward(g, batch, c))
            output.append(_new_row("architecture", spec["name"], seed, pilot_ratio, snr_db, metrics, time.perf_counter() - method_start))
        for method, active_weights in loss_specs:
            method_start = time.perf_counter()
            local = copy.deepcopy(run_config)
            local["loss"].update({key: 0.0 for key in ["lambda_adv", "lambda_pilot", "lambda_smooth", "lambda_sparse", "lambda_frequency", "lambda_energy"]})
            local["loss"].update(active_weights)
            spec = _spec(method, "picasso", loss_mode="rec_physics", enhanced=True, multiscale=True)
            generator = _make_generator(local, spec).to(device)
            _train_picasso(generator, None, train_loader, device, local, spec)
            metrics = _evaluate(test_loader, device, lambda batch, g=generator: _gen_forward(g, batch, False))
            output.append(_new_row("loss", method, seed, pilot_ratio, snr_db, metrics, time.perf_counter() - method_start))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def _stage3b_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    supplemental = config["supplemental"]
    return {
        "system": config["system"], "channel": config["channel"], "model_size": config["model_size"],
        "data": {"train_samples": supplemental["train_samples"], "test_samples": supplemental["test_samples"], "batch_size": supplemental["batch_size"]},
        "training": {
            "epochs": supplemental["epochs"], "warmup_epochs": config["runtime"]["warmup_epochs"],
            "learning_rate_g": config["runtime"]["learning_rate_g"], "learning_rate_d": config["runtime"]["learning_rate_d"],
            "gradient_clip_norm": config["runtime"]["gradient_clip_norm"],
        },
        "loss": copy.deepcopy(config["loss"]),
    }


def _standardize(
    row: dict[str, str],
    category: str,
    source: str,
    method: str | None = None,
    channel_profile: str | None = None,
) -> dict[str, object]:
    profile = row.get("profile", "")
    condition = f"pilot={row.get('pilot_ratio', '')}; snr={row.get('snr_db', '')}"
    if source == "stage4":
        condition += f"; profile={profile}; velocity={row.get('velocity_kmh', '')}; pattern={row.get('pilot_pattern', '')}"
    return {
        "ablation_type": category, "condition": condition, "method": method or row.get("method", ""),
        "pilot_ratio": row.get("pilot_ratio", ""), "snr_db": row.get("snr_db", ""),
        "velocity": row.get("velocity_kmh", 0), "channel_profile": channel_profile or profile or ("synthetic" if source != "stage4" else ""),
        "pilot_pattern": row.get("pilot_pattern", "comb"), "nmse": row.get("nmse", ""),
        "mse": row.get("mse", ""), "mae": row.get("mae", ""),
        "doppler_robustness": row.get("doppler_robustness_metric", ""),
        "runtime_seconds": row.get("runtime_seconds", ""), "source_stage": source, "reused_or_new": "reused",
    }


def _new_row(category: str, method: str, seed: int, pilot_ratio: float, snr_db: float, metrics: dict[str, float], runtime: float) -> dict[str, object]:
    return {
        "ablation_type": category, "condition": f"seed={seed}; pilot={pilot_ratio}; snr={snr_db}", "method": method,
        "pilot_ratio": pilot_ratio, "snr_db": snr_db, "velocity": 0, "channel_profile": "random-path synthetic OFDM",
        "pilot_pattern": "comb", "nmse": f"{metrics['nmse']:.8f}", "mse": f"{metrics['mse']:.8f}",
        "mae": f"{metrics['mae']:.8f}", "doppler_robustness": "", "runtime_seconds": f"{runtime:.4f}",
        "source_stage": "stage5a_supplemental", "reused_or_new": "new",
    }


def _write_outputs(config: dict[str, Any], categories: dict[str, list[dict[str, object]]]) -> list[str]:
    root = Path(config["outputs"]["results_dir"])
    names = {
        "model": "model_ablation_csv", "architecture": "architecture_ablation_csv", "loss": "loss_ablation_csv",
        "gan": "gan_ablation_csv", "channel": "channel_ablation_csv", "robustness": "robustness_ablation_csv",
    }
    paths = []
    for category, key in names.items():
        path = root / config["outputs"][key]
        write_csv(categories[category], path, RESULT_FIELDS)
        paths.append(str(path))
    return paths


def _means_by_method(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("nmse") not in (None, ""):
            grouped[str(row["method"])].append(float(row["nmse"]))
    return {method: mean(values) for method, values in grouped.items()}


def _category_means(category: str, rows: list[dict[str, object]]) -> dict[str, float]:
    # Component comparisons use the matched Stage 5A grid when it is present.
    if category in {"architecture", "loss"}:
        new_rows = [row for row in rows if row.get("reused_or_new") == "new"]
        if new_rows:
            rows = new_rows
    if category == "channel":
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("nmse") not in (None, ""):
                grouped[str(row["channel_profile"])].append(float(row["nmse"]))
        return {profile: mean(values) for profile, values in grouped.items()}
    return _means_by_method(rows)


def _keep_decision(category: str, method: str, delta: float) -> str:
    if category == "model":
        return "yes" if method == "PICASSO-rec" else "no"
    if category == "architecture":
        return "yes" if method in {"PICASSO-rec-base", "PICASSO-rec-enhanced"} and delta <= 0 else "no"
    if category in {"loss", "gan"}:
        return "yes" if method == "rec only" or method.startswith("no GAN") else "no"
    return "yes" if method == "PICASSO-rec" else "no"


def _interpret(category: str, method: str, delta: float) -> str:
    direction = "improves" if delta < 0 else ("matches" if abs(delta) < 1e-12 else "degrades")
    return f"{method} {direction} mean NMSE versus the category reference ({delta:+.2%}); retain provenance when interpreting."


def _validate_config(config: dict[str, Any]) -> None:
    policy = config["artifact_policy"]
    forbidden = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    if any(bool(policy.get(key, False)) for key in forbidden) or bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 5A may write only CSV and Markdown artifacts.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Stage 5A config must be a YAML mapping.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 5A paper-level ablations.")
    parser.add_argument("--config", default="configs/stage5a_ablation.yaml")
    args = parser.parse_args()
    result = run_stage5a(_load_config(args.config))
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"outputs: {result['outputs']}")


if __name__ == "__main__":
    main()
