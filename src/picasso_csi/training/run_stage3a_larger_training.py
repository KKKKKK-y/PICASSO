"""Run Stage 3A-L larger PICASSO training experiments."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import NoisySyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import (  # noqa: E402
    ResultTable,
    delay_domain_sparsity_score,
    mae,
    mse,
    nmse,
    pilot_consistency_error,
    write_csv,
)
from picasso_csi.losses import discriminator_bce_loss, picasso_generator_loss  # noqa: E402
from picasso_csi.models import DnCNNBaseline, EnhancedDnCNNBaseline, PICASSODiscriminator, PICASSOGenerator, ls_baseline  # noqa: E402


RAW_FIELDS = [
    "stage",
    "seed",
    "method",
    "pilot_ratio",
    "snr_db",
    "nmse",
    "mse",
    "mae",
    "pilot_consistency_error",
    "delay_sparsity_score",
    "epochs_used",
    "train_samples",
    "val_samples",
    "test_samples",
    "runtime_seconds",
    "loss_mode",
    "lambda_adv",
    "lambda_pilot",
    "lambda_smooth",
    "lambda_sparse",
    "param_count",
]

CURVE_FIELDS = ["stage", "seed", "method", "pilot_ratio", "snr_db", "epoch", "train_loss", "val_nmse", "learning_rate"]


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_stage3a_larger(config)
    print(f"device: {result['device']}")
    print(f"grid type: {result['scope']}")
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"best method: {result['best_method_overall']}")
    print(f"PICASSO-full beats Enhanced-DnCNN: {result['picasso_full_beats_enhanced']}")
    print(f"PICASSO-cond-full low SNR better than PICASSO-full: {result['cond_full_low_snr_better']}")
    print(f"checkpoint path local only: {result['checkpoint_dir']}")
    print("output files:")
    for path in result["output_files"]:
        print(f"  {path}")


def run_stage3a_larger(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    scope = _select_scope(config)
    device = _resolve_device(str(config["training"].get("device", "cuda")))
    table = ResultTable()
    curve_rows: list[dict[str, object]] = []
    methods = _method_specs(config)
    param_counts = _param_counts(config)

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"grid type: {scope['name']}")
    print(f"total combinations: {len(scope['seeds']) * len(scope['pilot_ratios']) * len(scope['snr_db_values'])}")
    print(f"PICASSO Generator parameters: {param_counts['PICASSOGenerator']}")
    print(f"PICASSO Discriminator parameters: {param_counts['PICASSODiscriminator']}")
    print(f"DnCNN parameters: {param_counts['DnCNN']}")
    print(f"Enhanced-DnCNN parameters: {param_counts['Enhanced-DnCNN']}")

    checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
    if bool(config["checkpoint"].get("save_local_checkpoint", False)):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for seed in scope["seeds"]:
        _set_seed(int(seed))
        for pilot_ratio in scope["pilot_ratios"]:
            for snr_db in scope["snr_db_values"]:
                print(f"current seed: {seed}, current pilot ratio: {pilot_ratio}, current SNR: {snr_db}")
                train_loader, val_loader, test_loader = _build_loaders(config, scope, int(seed), float(pilot_ratio), float(snr_db))
                for spec in methods:
                    method_start = time.perf_counter()
                    print(f"current method: {spec['name']}")
                    metrics, epochs_used, curves, checkpoint_path = _run_method(
                        spec, train_loader, val_loader, test_loader, device, config, scope, int(seed), float(pilot_ratio), float(snr_db)
                    )
                    curve_rows.extend(curves)
                    runtime_seconds = time.perf_counter() - method_start
                    table.append(
                        _make_row(config, scope, spec, int(seed), float(pilot_ratio), float(snr_db), metrics, epochs_used, runtime_seconds, param_counts[spec["param_key"]])
                    )
                    print(f"test NMSE: {metrics['nmse']:.6f}")
                    if checkpoint_path:
                        print(f"checkpoint path local only: {checkpoint_path}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    runtime_seconds = time.perf_counter() - start
    output_files = _save_outputs(config, table, curve_rows, scope, runtime_seconds, param_counts)
    summary = table.grouped_summary()
    print("mean NMSE by method:")
    for row in summary:
        print(f"  {row['method']}: {row['nmse_mean']}")
    result = _summarize_result(table)
    result.update(
        {
            "device": str(device),
            "scope": scope["name"],
            "runtime_seconds": runtime_seconds,
            "output_files": output_files,
            "checkpoint_dir": str(checkpoint_dir),
            "param_counts": param_counts,
        }
    )
    return result


def _run_method(
    spec: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    scope: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
) -> tuple[dict[str, float], int, list[dict[str, object]], str]:
    if spec["kind"] == "ls":
        metrics = _evaluate_estimator(test_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))
        return metrics, 0, [], ""

    if spec["kind"] == "dncnn":
        model = _make_dncnn(config, enhanced=spec["enhanced"]).to(device)
        estimator = lambda batch: model(batch["H_sparse"], batch["mask"])
        epochs, curves, checkpoint_path = _train_supervised(
            model, train_loader, val_loader, estimator, device, config, scope, spec, seed, pilot_ratio, snr_db
        )
        metrics = _evaluate_estimator(test_loader, device, estimator)
        return metrics, epochs, curves, checkpoint_path

    generator = _make_generator(config, use_condition=spec["use_condition"]).to(device)
    discriminator = _make_discriminator(config).to(device) if spec["uses_adv"] else None
    estimator = lambda batch: _generator_forward(generator, batch, spec["use_condition"])
    epochs, curves, checkpoint_path = _train_picasso(
        generator, discriminator, train_loader, val_loader, estimator, device, config, scope, spec, seed, pilot_ratio, snr_db
    )
    metrics = _evaluate_estimator(test_loader, device, estimator)
    return metrics, epochs, curves, checkpoint_path


def _train_supervised(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    scope: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
) -> tuple[int, list[dict[str, object]], str]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    best_val = float("inf")
    stale = 0
    curves = []
    checkpoint_path = ""
    for epoch in range(1, int(scope["epochs"]) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(estimator(batch), batch["H_full"])
            loss.backward()
            _clip(model, config)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_nmse = _evaluate_estimator(val_loader, device, estimator)["nmse"]
        curves.append(_curve_row(config, spec, seed, pilot_ratio, snr_db, epoch, mean(losses), val_nmse, optimizer.param_groups[0]["lr"]))
        print(f"  epoch {epoch}: train loss {mean(losses):.6f}, val NMSE {val_nmse:.6f}")
        if val_nmse + 1e-7 < best_val:
            best_val = val_nmse
            stale = 0
            checkpoint_path = _save_checkpoint(model, config, spec, seed, pilot_ratio, snr_db)
        else:
            stale += 1
        if stale >= int(scope["early_stop_patience"]):
            return epoch, curves, checkpoint_path
    return int(scope["epochs"]), curves, checkpoint_path


def _train_picasso(
    generator: PICASSOGenerator,
    discriminator: PICASSODiscriminator | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    device: torch.device,
    config: dict[str, Any],
    scope: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
) -> tuple[int, list[dict[str, object]], str]:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["learning_rate_g"]))
    optimizer_d = (
        torch.optim.Adam(discriminator.parameters(), lr=float(config["training"]["learning_rate_d"]))
        if discriminator is not None
        else None
    )
    loss_config = config["loss"]
    best_val = float("inf")
    stale = 0
    curves = []
    checkpoint_path = ""
    for epoch in range(1, int(scope["epochs"]) + 1):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        losses = []
        active_mode = _active_loss_mode(spec, config, epoch, scope)
        for batch in train_loader:
            batch = _to_device(batch, device)
            if discriminator is not None and optimizer_d is not None and active_mode == "full":
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    H_fake = _generator_forward(generator, batch, spec["use_condition"])
                d_loss = discriminator_bce_loss(discriminator(batch["H_full"]), discriminator(H_fake.detach()))
                d_loss.backward()
                _clip(discriminator, config)
                optimizer_d.step()
            optimizer_g.zero_grad(set_to_none=True)
            H_hat = _generator_forward(generator, batch, spec["use_condition"])
            fake_logits = discriminator(H_hat) if discriminator is not None and active_mode == "full" else None
            loss = picasso_generator_loss(
                H_hat,
                batch["H_full"],
                batch["H_sparse"],
                batch["mask"],
                fake_logits=fake_logits,
                lambda_rec=float(loss_config["lambda_rec"]),
                lambda_adv=float(loss_config["lambda_adv"]),
                lambda_pilot=float(loss_config["lambda_pilot"]),
                lambda_smooth=float(loss_config["lambda_smooth"]),
                lambda_sparse=float(loss_config["lambda_sparse"]),
                loss_mode=active_mode,
            )["total"]
            loss.backward()
            _clip(generator, config)
            optimizer_g.step()
            losses.append(float(loss.detach().cpu()))
        val_nmse = _evaluate_estimator(val_loader, device, estimator)["nmse"]
        curves.append(_curve_row(config, spec, seed, pilot_ratio, snr_db, epoch, mean(losses), val_nmse, optimizer_g.param_groups[0]["lr"]))
        print(f"  epoch {epoch}: train loss {mean(losses):.6f}, val NMSE {val_nmse:.6f}")
        if val_nmse + 1e-7 < best_val:
            best_val = val_nmse
            stale = 0
            checkpoint_path = _save_checkpoint(generator, config, spec, seed, pilot_ratio, snr_db)
        else:
            stale += 1
        if stale >= int(scope["early_stop_patience"]):
            return epoch, curves, checkpoint_path
    return int(scope["epochs"]), curves, checkpoint_path


@torch.no_grad()
def _evaluate_estimator(loader: DataLoader, device: torch.device, estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor]) -> dict[str, float]:
    totals = {key: 0.0 for key in ["samples", "nmse", "mse", "mae", "pilot_consistency_error", "delay_sparsity_score"]}
    for batch in loader:
        batch = _to_device(batch, device)
        H_hat = estimator(batch)
        n = float(batch["H_full"].shape[0])
        totals["samples"] += n
        totals["nmse"] += float(nmse(H_hat, batch["H_full"]).detach().cpu()) * n
        totals["mse"] += float(mse(H_hat, batch["H_full"]).detach().cpu()) * n
        totals["mae"] += float(mae(H_hat, batch["H_full"]).detach().cpu()) * n
        totals["pilot_consistency_error"] += float(pilot_consistency_error(H_hat, batch["H_sparse"], batch["mask"]).detach().cpu()) * n
        totals["delay_sparsity_score"] += float(delay_domain_sparsity_score(H_hat).detach().cpu()) * n
    samples = max(totals["samples"], 1.0)
    return {key: value / samples for key, value in totals.items() if key != "samples"}


def _build_loaders(config: dict[str, Any], scope: dict[str, Any], seed: int, pilot_ratio: float, snr_db: float) -> tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = int(config["data"]["batch_size"])
    return (
        DataLoader(_build_dataset(config, scope["train_samples"], seed, pilot_ratio, snr_db), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(_build_dataset(config, scope["val_samples"], seed + 50_000, pilot_ratio, snr_db), batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(_build_dataset(config, scope["test_samples"], seed + 100_000, pilot_ratio, snr_db), batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _build_dataset(config: dict[str, Any], num_samples: int, seed: int, pilot_ratio: float, snr_db: float) -> NoisySyntheticCSIDataset:
    system = config["system"]
    channel = config["channel"]
    return NoisySyntheticCSIDataset(
        num_samples=int(num_samples),
        n_tx=int(system["n_tx"]),
        n_rx=int(system["n_rx"]),
        n_subcarriers=int(system["n_subcarriers"]),
        n_paths=int(system["n_paths"]),
        pilot_ratio=pilot_ratio,
        snr_db=snr_db,
        seed=seed,
        random_n_paths=bool(channel["random_n_paths"]),
        min_paths=int(channel["min_paths"]),
        max_paths=int(channel["max_paths"]),
        delay_spread=channel["delay_spread"],
        gain_distribution=str(channel["gain_distribution"]),
        normalize_channel=bool(channel["normalize_channel"]),
        pilot_noise_only=bool(channel["pilot_noise_only"]),
    )


def _make_dncnn(config: dict[str, Any], enhanced: bool) -> DnCNNBaseline:
    if enhanced:
        return EnhancedDnCNNBaseline(
            hidden_channels=int(config["model_size"]["enhanced_dncnn_base_channels"]),
            depth=int(config["model_size"]["enhanced_dncnn_num_blocks"]),
        )
    return DnCNNBaseline()


def _make_generator(config: dict[str, Any], use_condition: bool) -> PICASSOGenerator:
    return PICASSOGenerator(
        base_channels=int(config["model_size"]["generator_base_channels"]),
        num_blocks=int(config["model_size"]["generator_num_blocks"]),
        use_condition=use_condition,
    )


def _make_discriminator(config: dict[str, Any]) -> PICASSODiscriminator:
    return PICASSODiscriminator(base_channels=int(config["model_size"]["discriminator_base_channels"]))


def _generator_forward(generator: PICASSOGenerator, batch: dict[str, torch.Tensor], use_condition: bool) -> torch.Tensor:
    if use_condition:
        return generator(batch["H_sparse"], batch["snr_db"], batch["pilot_ratio"])
    return generator(batch["H_sparse"])


def _method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config["models"]
    specs = []
    if models.get("run_ls", True):
        specs.append(_spec("LS", "ls", "", False, False, False, "LS"))
    if models.get("run_dncnn", True):
        specs.append(_spec("DnCNN", "dncnn", "", False, False, False, "DnCNN"))
    if models.get("run_enhanced_dncnn", True):
        specs.append(_spec("Enhanced-DnCNN", "dncnn", "", False, False, True, "Enhanced-DnCNN"))
    if models.get("run_picasso_rec", True):
        specs.append(_spec("PICASSO-rec", "picasso", "rec_only", False, False, False, "PICASSOGenerator"))
    if models.get("run_picasso_rec_physics", True):
        specs.append(_spec("PICASSO-rec-physics", "picasso", "rec_physics", False, False, False, "PICASSOGenerator"))
    if models.get("run_picasso_full", True):
        specs.append(_spec("PICASSO-full", "picasso", "full", True, False, False, "PICASSOGenerator"))
    if models.get("run_picasso_cond_full", True):
        specs.append(_spec("PICASSO-cond-full", "picasso", "full", True, True, False, "PICASSOGeneratorCond"))
    return specs


def _spec(name: str, kind: str, loss_mode: str, uses_adv: bool, use_condition: bool, enhanced: bool, param_key: str) -> dict[str, Any]:
    return {"name": name, "kind": kind, "loss_mode": loss_mode, "uses_adv": uses_adv, "use_condition": use_condition, "enhanced": enhanced, "param_key": param_key}


def _param_counts(config: dict[str, Any]) -> dict[str, int]:
    return {
        "LS": 0,
        "DnCNN": _count_parameters(DnCNNBaseline()),
        "Enhanced-DnCNN": _count_parameters(_make_dncnn(config, enhanced=True)),
        "PICASSOGenerator": _count_parameters(_make_generator(config, use_condition=False)),
        "PICASSOGeneratorCond": _count_parameters(_make_generator(config, use_condition=True)),
        "PICASSODiscriminator": _count_parameters(_make_discriminator(config)),
    }


def _select_scope(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    training = config["training"]
    diagnosis = config["diagnosis"]
    estimated_units = len(data["seeds"]) * len(data["pilot_ratios"]) * len(data["snr_db_values"]) * len(_method_specs(config)) * int(data["train_samples"]) * int(training["epochs"])
    use_reduced = bool(training["allow_reduced_grid_if_runtime_high"]) and estimated_units > 40_000_000
    if not use_reduced:
        return {
            "name": "full",
            "reason": "full grid estimate is within the runtime budget",
            "seeds": list(data["seeds"]),
            "pilot_ratios": list(data["pilot_ratios"]),
            "snr_db_values": list(data["snr_db_values"]),
            "train_samples": int(data["train_samples"]),
            "val_samples": int(data["val_samples"]),
            "test_samples": int(data["test_samples"]),
            "epochs": int(training["epochs"]),
            "warmup_epochs": int(training["warmup_epochs"]),
            "early_stop_patience": int(training["early_stop_patience"]),
        }
    return {
        "name": "reduced",
        "reason": f"full grid estimated units {estimated_units} exceeded the 5-hour diagnostic budget",
        "seeds": list(diagnosis["reduced_grid_seeds"]),
        "pilot_ratios": list(diagnosis["reduced_grid_pilot_ratios"]),
        "snr_db_values": list(diagnosis["reduced_grid_snr_db_values"]),
        "train_samples": int(diagnosis["reduced_train_samples"]),
        "val_samples": int(diagnosis["reduced_val_samples"]),
        "test_samples": int(diagnosis["reduced_test_samples"]),
        "epochs": int(diagnosis["reduced_epochs"]),
        "warmup_epochs": int(diagnosis["reduced_warmup_epochs"]),
        "early_stop_patience": int(diagnosis["reduced_early_stop_patience"]),
    }


def _save_outputs(config: dict[str, Any], table: ResultTable, curve_rows: list[dict[str, object]], scope: dict[str, Any], runtime_seconds: float, param_counts: dict[str, int]) -> list[str]:
    outputs = config["outputs"]
    results_dir = Path(outputs["results_dir"])
    raw = write_csv(table.rows, results_dir / outputs["raw_csv"], RAW_FIELDS)
    summary = table.save_summary_by_condition_csv(results_dir / outputs["summary_csv"])
    paper = table.save_paper_table_csv(results_dir / outputs["paper_table_csv"])
    curves = write_csv(curve_rows, results_dir / outputs["training_curve_csv"], CURVE_FIELDS)
    diagnosis = table.save_diagnosis_csv(results_dir / outputs["diagnosis_csv"])
    analysis = _write_analysis(config, table, scope, runtime_seconds, param_counts)
    return [str(path) for path in [raw, summary, paper, curves, diagnosis, analysis]]


def _write_analysis(config: dict[str, Any], table: ResultTable, scope: dict[str, Any], runtime_seconds: float, param_counts: dict[str, int]) -> Path:
    path = Path(config["outputs"]["analysis_md"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result = _summarize_result(table)
    summary = table.grouped_summary()
    lines = [
        "# Stage 3A-L Larger PICASSO Training",
        "",
        "## Why Larger Models And Longer Training",
        "Stage 2BC and Stage 3A showed promising PICASSO trends under short diagnostics. Stage 3A-L increases model capacity and trains longer to test whether those gains survive a controlled larger-budget run.",
        "",
        "## Parameter Counts",
        f"- DnCNN: {param_counts['DnCNN']}",
        f"- Enhanced-DnCNN: {param_counts['Enhanced-DnCNN']}",
        f"- PICASSO Generator: {param_counts['PICASSOGenerator']}",
        f"- PICASSO Conditional Generator: {param_counts['PICASSOGeneratorCond']}",
        f"- PICASSO Discriminator: {param_counts['PICASSODiscriminator']}",
        "",
        "## Training Budget",
        f"- Grid type: {scope['name']}",
        f"- Reason: {scope['reason']}",
        f"- Runtime seconds: {runtime_seconds:.2f}",
        f"- Train/val/test samples: {scope['train_samples']} / {scope['val_samples']} / {scope['test_samples']}",
        f"- Epochs: {scope['epochs']}",
        "",
        "## Data And Loss Setup",
        "- Random path count, random delay spread, normalized synthetic channels, and pilot-only AWGN are enabled.",
        f"- Loss weights: adv={config['loss']['lambda_adv']}, pilot={config['loss']['lambda_pilot']}, smooth={config['loss']['lambda_smooth']}, sparse={config['loss']['lambda_sparse']}.",
        f"- GAN warmup: {scope['warmup_epochs']} epochs with rec_physics before light adversarial training.",
        "",
        "## Overall Results",
    ]
    for row in summary:
        lines.append(f"- {row['method']}: NMSE {row['nmse_mean']} +/- {row['nmse_std']} over {row['count']} rows")
    lines.extend(
        [
            "",
            "## Low Pilot Ratio Results",
            f"Best method for pilot_ratio <= 0.125: {result['best_method_low_pilot']}.",
            "",
            "## Low SNR Results",
            f"Best method for SNR <= 10 dB: {result['best_method_low_snr']}.",
            "",
            "## PICASSO-Full vs Enhanced-DnCNN",
            f"PICASSO-full beats Enhanced-DnCNN: {result['picasso_full_beats_enhanced']}.",
            "",
            "## Condition-Aware Low SNR",
            f"PICASSO-cond-full is better than PICASSO-full at low SNR: {result['cond_full_low_snr_better']}.",
            "",
            "## Stability And Adversarial Loss",
            f"PICASSO-rec-physics stable over Enhanced-DnCNN: {result['picasso_physics_beats_enhanced']}.",
            f"Adversarial improvement over PICASSO-rec-physics: {result['adv_beats_physics']}.",
            "",
            "## Stage 3B Recommendation",
            _stage3b_recommendation(result),
            "",
            "## 3GPP CDL / QuaDRiGa",
            "A larger synthetic diagnostic is still not enough for final paper claims. Stage 3B should introduce 3GPP CDL or QuaDRiGa before scaling formal experiments.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summarize_result(table: ResultTable) -> dict[str, object]:
    means = {row["method"]: float(row["nmse_mean"]) for row in table.grouped_summary()}
    full_low = _conditional_mean(table, "PICASSO-full", lambda row: float(row["snr_db"]) <= 10)
    cond_low = _conditional_mean(table, "PICASSO-cond-full", lambda row: float(row["snr_db"]) <= 10)
    return {
        "best_method_overall": min(means, key=means.get) if means else "",
        "best_method_low_pilot": table.winner(lambda row: float(row["pilot_ratio"]) <= 0.125),
        "best_method_low_snr": table.winner(lambda row: float(row["snr_db"]) <= 10),
        "picasso_full_beats_enhanced": bool("PICASSO-full" in means and "Enhanced-DnCNN" in means and means["PICASSO-full"] < means["Enhanced-DnCNN"]),
        "picasso_physics_beats_enhanced": bool("PICASSO-rec-physics" in means and "Enhanced-DnCNN" in means and means["PICASSO-rec-physics"] < means["Enhanced-DnCNN"]),
        "adv_beats_physics": bool("PICASSO-full" in means and "PICASSO-rec-physics" in means and means["PICASSO-full"] < means["PICASSO-rec-physics"]),
        "cond_full_low_snr_better": bool(cond_low is not None and full_low is not None and cond_low < full_low),
    }


def _stage3b_recommendation(result: dict[str, object]) -> str:
    if result["picasso_full_beats_enhanced"] and result["adv_beats_physics"]:
        return "Proceed to Stage 3B with realistic channel models and retain full PICASSO as a main candidate."
    if result["picasso_physics_beats_enhanced"] and not result["adv_beats_physics"]:
        return "Scale supervised/physics-guided PICASSO first; keep adversarial loss as an ablation until it proves useful."
    return "Improve supervised reconstruction and channel realism before a full-scale GAN experiment."


def _conditional_mean(table: ResultTable, method: str, predicate) -> float | None:
    values = [float(row["nmse"]) for row in table.rows if row["method"] == method and predicate(row)]
    return mean(values) if values else None


def _active_loss_mode(spec: dict[str, Any], config: dict[str, Any], epoch: int, scope: dict[str, Any]) -> str:
    if not spec["uses_adv"]:
        return str(spec["loss_mode"])
    if bool(config["gan_schedule"]["use_warmup"]) and epoch <= int(scope["warmup_epochs"]):
        return str(config["gan_schedule"]["warmup_loss_mode"])
    return "full"


def _make_row(config: dict[str, Any], scope: dict[str, Any], spec: dict[str, Any], seed: int, pilot_ratio: float, snr_db: float, metrics: dict[str, float], epochs_used: int, runtime_seconds: float, param_count: int) -> dict[str, object]:
    loss = config["loss"]
    has_physics = "physics" in spec["name"] or spec["uses_adv"]
    return {
        "stage": config["project"]["stage"],
        "seed": seed,
        "method": spec["name"],
        "pilot_ratio": f"{pilot_ratio:.4f}",
        "snr_db": f"{snr_db:g}",
        "nmse": f"{metrics['nmse']:.8f}",
        "mse": f"{metrics['mse']:.8f}",
        "mae": f"{metrics['mae']:.8f}",
        "pilot_consistency_error": f"{metrics['pilot_consistency_error']:.8f}",
        "delay_sparsity_score": f"{metrics['delay_sparsity_score']:.8f}",
        "epochs_used": epochs_used,
        "train_samples": scope["train_samples"],
        "val_samples": scope["val_samples"],
        "test_samples": scope["test_samples"],
        "runtime_seconds": f"{runtime_seconds:.4f}",
        "loss_mode": spec["loss_mode"],
        "lambda_adv": loss["lambda_adv"] if spec["uses_adv"] else 0.0,
        "lambda_pilot": loss["lambda_pilot"] if has_physics else 0.0,
        "lambda_smooth": loss["lambda_smooth"] if has_physics else 0.0,
        "lambda_sparse": loss["lambda_sparse"] if has_physics else 0.0,
        "param_count": param_count,
    }


def _curve_row(config: dict[str, Any], spec: dict[str, Any], seed: int, pilot_ratio: float, snr_db: float, epoch: int, train_loss: float, val_nmse: float, learning_rate: float) -> dict[str, object]:
    return {
        "stage": config["project"]["stage"],
        "seed": seed,
        "method": spec["name"],
        "pilot_ratio": f"{pilot_ratio:.4f}",
        "snr_db": f"{snr_db:g}",
        "epoch": epoch,
        "train_loss": f"{train_loss:.8f}",
        "val_nmse": f"{val_nmse:.8f}",
        "learning_rate": f"{learning_rate:.8f}",
    }


def _save_checkpoint(model: torch.nn.Module, config: dict[str, Any], spec: dict[str, Any], seed: int, pilot_ratio: float, snr_db: float) -> str:
    if not bool(config["checkpoint"].get("save_local_checkpoint", False)):
        return ""
    checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_method = str(spec["name"]).replace("/", "_").replace(" ", "_")
    path = checkpoint_dir / f"{safe_method}_seed{seed}_p{pilot_ratio:g}_snr{snr_db:g}.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def _clip(model: torch.nn.Module, config: dict[str, Any]) -> None:
    max_norm = float(config["training"].get("gradient_clip_norm", 0.0))
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 3A-L must not save generated datasets.")
    if bool(config["checkpoint"].get("commit_checkpoints", False)):
        raise ValueError("Stage 3A-L checkpoints must be local only.")
    policy = config["artifact_policy"]
    if bool(policy.get("commit_checkpoints", False)) or bool(policy.get("save_numpy_arrays", False)):
        raise ValueError("Stage 3A-L must not commit checkpoints or save numpy arrays.")


def _resolve_device(requested_device: str) -> torch.device:
    return torch.device("cuda" if requested_device.lower() == "cuda" and torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path!r} must contain a YAML mapping.")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3A-L larger PICASSO training.")
    parser.add_argument("--config", default="configs/stage3a_larger_training.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
