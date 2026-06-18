"""Run Stage 4 CDL-inspired realistic channel generalization experiments."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import CDLChannelDataset  # noqa: E402
from picasso_csi.evaluation import (  # noqa: E402
    delay_domain_sparsity_score,
    doppler_robustness_metric,
    mae,
    mse,
    nmse,
    pilot_consistency_error,
    write_csv,
)
from picasso_csi.losses import picasso_generator_loss  # noqa: E402
from picasso_csi.models import EnhancedDnCNNBaseline, PICASSOGenerator, lmmse_like_baseline, ls_baseline  # noqa: E402


RAW_FIELDS = [
    "stage",
    "method",
    "profile",
    "pilot_pattern",
    "seed",
    "pilot_ratio",
    "snr_db",
    "velocity_kmh",
    "nmse",
    "mse",
    "mae",
    "pilot_consistency_error",
    "delay_domain_sparsity_score",
    "doppler_robustness_metric",
    "runtime_seconds",
]


def main() -> None:
    result = run_stage4(_load_config(_parse_args().config))
    print(f"device: {result['device']}")
    print(f"grid type: {result['grid_type']}")
    print(f"rows: {result['rows']}")
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"best method: {result['best_method']}")
    print(f"outputs: {result['outputs']}")


def run_stage4(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    device = _resolve_device(config["training"]["device"])
    scope = _scope(config)
    rows: list[dict[str, object]] = []
    print(f"device: {device}")
    print(f"grid type: {scope['name']}")
    print(f"reduction reason: {scope['reason']}")
    print(f"profiles: {scope['profiles']}")
    print(f"seeds: {scope['seeds']}, velocities: {scope['velocities_kmh']}, SNR: {scope['snr_db_values']}")

    for profile in scope["profiles"]:
        for seed in scope["seeds"]:
            _set_seed(int(seed))
            for pilot_ratio in scope["pilot_ratios"]:
                for snr_db in scope["snr_db_values"]:
                    for velocity in scope["velocities_kmh"]:
                        pattern = _pilot_pattern(config, pilot_ratio)
                        train_loader, test_loader = _loaders(config, profile, pattern, int(seed), float(pilot_ratio), float(snr_db), float(velocity))
                        print(f"profile={profile}, seed={seed}, pilot={pilot_ratio}, snr={snr_db}, velocity={velocity}, pattern={pattern}")
                        for method in _methods(config):
                            method_start = time.perf_counter()
                            metrics = _run_method(method, train_loader, test_loader, device, config)
                            rows.append(_row(config, method, profile, pattern, int(seed), float(pilot_ratio), float(snr_db), float(velocity), metrics, time.perf_counter() - method_start))
                            print(f"  {method}: NMSE {metrics['nmse']:.6f}")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    outputs = _save_outputs(config, rows, scope, time.perf_counter() - start)
    means = _means(rows)
    return {
        "device": str(device),
        "grid_type": scope["name"],
        "rows": len(rows),
        "runtime_seconds": time.perf_counter() - start,
        "best_method": min(means, key=means.get),
        "outputs": outputs,
    }


def _run_method(method: str, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    if method == "LS":
        return _evaluate(test_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))
    if method == "LMMSE-like":
        return _evaluate(test_loader, device, lambda batch: lmmse_like_baseline(batch["H_sparse"], batch["mask"]))
    if method == "Enhanced-DnCNN":
        model = EnhancedDnCNNBaseline(hidden_channels=64, depth=8).to(device)
        _train_supervised(model, train_loader, device, config, lambda batch: model(batch["H_sparse"], batch["mask"]))
        return _evaluate(test_loader, device, lambda batch: model(batch["H_sparse"], batch["mask"]))
    generator = PICASSOGenerator(base_channels=64, num_blocks=6).to(device)
    loss_mode = "rec_physics" if method == "PICASSO-rec-physics" else "rec_only"
    _train_picasso(generator, train_loader, device, config, loss_mode)
    return _evaluate(test_loader, device, lambda batch: generator(batch["H_sparse"]))


def _train_supervised(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: dict[str, Any], estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor]) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    for _ in range(int(config["training"]["epochs"])):
        for batch in loader:
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(estimator(batch), batch["H_full"])
            loss.backward()
            _clip(model, config)
            opt.step()


def _train_picasso(generator: PICASSOGenerator, loader: DataLoader, device: torch.device, config: dict[str, Any], loss_mode: str) -> None:
    opt = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["learning_rate"]))
    loss_cfg = config["loss"]
    for _ in range(int(config["training"]["epochs"])):
        for batch in loader:
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            H_hat = generator(batch["H_sparse"])
            loss = picasso_generator_loss(
                H_hat,
                batch["H_full"],
                batch["H_sparse"],
                batch["mask"],
                lambda_rec=float(loss_cfg["lambda_rec"]),
                lambda_pilot=float(loss_cfg["lambda_pilot"]),
                lambda_smooth=float(loss_cfg["lambda_smooth"]),
                lambda_sparse=float(loss_cfg["lambda_sparse"]),
                lambda_frequency=float(loss_cfg["lambda_frequency"]),
                lambda_energy=float(loss_cfg["lambda_energy"]),
                loss_mode=loss_mode,
            )["total"]
            loss.backward()
            _clip(generator, config)
            opt.step()


@torch.no_grad()
def _evaluate(loader: DataLoader, device: torch.device, estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor]) -> dict[str, float]:
    totals = defaultdict(float)
    for batch in loader:
        batch = _to_device(batch, device)
        H_hat = estimator(batch)
        n = float(batch["H_full"].shape[0])
        totals["samples"] += n
        totals["nmse"] += float(nmse(H_hat, batch["H_full"]).cpu()) * n
        totals["mse"] += float(mse(H_hat, batch["H_full"]).cpu()) * n
        totals["mae"] += float(mae(H_hat, batch["H_full"]).cpu()) * n
        totals["pilot_consistency_error"] += float(pilot_consistency_error(H_hat, batch["H_sparse"], batch["mask"]).cpu()) * n
        totals["delay_domain_sparsity_score"] += float(delay_domain_sparsity_score(H_hat).cpu()) * n
        totals["doppler_robustness_metric"] += float(doppler_robustness_metric(H_hat, batch["H_full"], batch["H_prev"]).cpu()) * n
    samples = max(totals["samples"], 1.0)
    return {key: totals[key] / samples for key in RAW_FIELDS if key in totals and key != "samples"}


def _loaders(config: dict[str, Any], profile: str, pattern: str, seed: int, pilot_ratio: float, snr_db: float, velocity: float) -> tuple[DataLoader, DataLoader]:
    batch_size = int(config["data"]["batch_size"])
    train = _dataset(config, int(config["data"]["train_samples"]), profile, pattern, seed, pilot_ratio, snr_db, velocity)
    test = _dataset(config, int(config["data"]["test_samples"]), profile, pattern, seed + 100_000, pilot_ratio, snr_db, velocity)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def _dataset(config: dict[str, Any], samples: int, profile: str, pattern: str, seed: int, pilot_ratio: float, snr_db: float, velocity: float) -> CDLChannelDataset:
    system = config["system"]
    return CDLChannelDataset(
        num_samples=samples,
        n_tx=int(system["n_tx"]),
        n_rx=int(system["n_rx"]),
        n_subcarriers=int(system["n_subcarriers"]),
        profile=profile,
        pilot_ratio=pilot_ratio,
        pilot_pattern=pattern,
        snr_db=snr_db,
        velocity_kmh=velocity,
        seed=seed,
        delay_spread_scale=config["cdl"]["delay_spread_scale"],
        pilot_contamination_std=float(config["cdl"]["pilot_contamination_std"]),
    )


def _scope(config: dict[str, Any]) -> dict[str, Any]:
    full_units = (
        len(config["cdl"]["profiles"])
        * len(config["data"]["seeds"])
        * len(config["data"]["pilot_ratios"])
        * len(config["data"]["snr_db_values"])
        * len(config["data"]["velocities_kmh"])
        * len(_methods(config))
        * int(config["data"]["train_samples"])
        * int(config["training"]["epochs"])
    )
    if bool(config["training"]["allow_reduced_grid_if_runtime_high"]) and full_units > 1_500_000:
        return {
            "name": "reduced",
            "reason": f"estimated units {full_units} exceeded threshold",
            "profiles": list(config["cdl"]["profiles"]),
            "seeds": list(config["reduction"]["seeds"]),
            "pilot_ratios": list(config["data"]["pilot_ratios"]),
            "snr_db_values": list(config["reduction"]["snr_db_values"]),
            "velocities_kmh": list(config["reduction"]["velocities_kmh"]),
        }
    return {
        "name": "full",
        "reason": "full grid within threshold",
        "profiles": list(config["cdl"]["profiles"]),
        "seeds": list(config["data"]["seeds"]),
        "pilot_ratios": list(config["data"]["pilot_ratios"]),
        "snr_db_values": list(config["data"]["snr_db_values"]),
        "velocities_kmh": list(config["data"]["velocities_kmh"]),
    }


def _methods(config: dict[str, Any]) -> list[str]:
    out = []
    if config["models"].get("run_ls", True):
        out.append("LS")
    if config["models"].get("run_lmmse", True):
        out.append("LMMSE-like")
    if config["models"].get("run_enhanced_dncnn", True):
        out.append("Enhanced-DnCNN")
    if config["models"].get("run_picasso_rec", True):
        out.append("PICASSO-rec")
    if config["models"].get("run_picasso_rec_physics", True):
        out.append("PICASSO-rec-physics")
    return out


def _pilot_pattern(config: dict[str, Any], pilot_ratio: float) -> str:
    patterns = list(config["cdl"]["pilot_patterns"])
    return patterns[int(round(1.0 / pilot_ratio)) % len(patterns)]


def _save_outputs(config: dict[str, Any], rows: list[dict[str, object]], scope: dict[str, Any], runtime_seconds: float) -> list[str]:
    out_cfg = config["outputs"]
    result_dir = Path(out_cfg["results_dir"])
    raw_path = result_dir / out_cfg["raw_csv"]
    summary_path = result_dir / out_cfg["summary_csv"]
    paper_path = result_dir / out_cfg["paper_table_csv"]
    diagnosis_path = Path(out_cfg["diagnosis_md"])
    report_path = Path(out_cfg["report_md"])
    write_csv(rows, raw_path, RAW_FIELDS)
    write_csv(_summary(rows), summary_path, ["method", "profile", "velocity_kmh", "nmse_mean", "nmse_std", "doppler_mean", "count"])
    write_csv(_paper(rows), paper_path, ["profile", "pilot_ratio", "snr_db", "velocity_kmh", *sorted({str(r["method"]) for r in rows}), "best_method"])
    text = _report(rows, scope, runtime_seconds)
    diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosis_path.write_text(text, encoding="utf-8")
    report_path.write_text(text, encoding="utf-8")
    return [str(raw_path), str(summary_path), str(paper_path), str(diagnosis_path), str(report_path)]


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["profile"], row["velocity_kmh"])].append(row)
    out = []
    for (method, profile, velocity), group in sorted(grouped.items()):
        nmse_values = [float(r["nmse"]) for r in group]
        doppler_values = [float(r["doppler_robustness_metric"]) for r in group]
        out.append({
            "method": method,
            "profile": profile,
            "velocity_kmh": velocity,
            "nmse_mean": f"{mean(nmse_values):.8f}",
            "nmse_std": f"{stdev(nmse_values) if len(nmse_values) > 1 else 0.0:.8f}",
            "doppler_mean": f"{mean(doppler_values):.8f}",
            "count": len(group),
        })
    return out


def _paper(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    methods = sorted({str(r["method"]) for r in rows})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["profile"], row["pilot_ratio"], row["snr_db"], row["velocity_kmh"])].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        record = {"profile": key[0], "pilot_ratio": key[1], "snr_db": key[2], "velocity_kmh": key[3]}
        best_method, best_value = "", float("inf")
        for method in methods:
            values = [float(r["nmse"]) for r in group if r["method"] == method]
            record[method] = f"{mean(values):.6f}" if values else ""
            if values and mean(values) < best_value:
                best_method, best_value = method, mean(values)
        record["best_method"] = best_method
        out.append(record)
    return out


def _report(rows: list[dict[str, object]], scope: dict[str, Any], runtime_seconds: float) -> str:
    means = _means(rows)
    low_velocity = _filtered_means(rows, lambda row: float(row["velocity_kmh"]) == 0.0)
    high_velocity = _filtered_means(rows, lambda row: float(row["velocity_kmh"]) >= 60.0)
    lines = [
        "# Stage 4 CDL Generalization Report",
        "",
        f"Grid type: {scope['name']}",
        f"Reduction: {scope['reason']}",
        f"Runtime seconds: {runtime_seconds:.2f}",
        "",
        "## CDL vs Synthetic Comparison",
        "Stage 4 replaces the earlier simple synthetic multipath generator with a CDL-inspired clustered channel model. The new data includes CDL-A/B/C profiles, spatial steering, delay-spread variation, pilot contamination noise, pilot patterns, and velocity-dependent Doppler phase.",
        "",
        "## Overall NMSE",
    ]
    for method, value in sorted(means.items(), key=lambda item: item[1]):
        lines.append(f"- {method}: {value:.8f}")
    lines.extend([
        "",
        "## Mobility Impact",
        f"- Best static-channel method: {min(low_velocity, key=low_velocity.get) if low_velocity else 'n/a'}",
        f"- Best high-mobility method: {min(high_velocity, key=high_velocity.get) if high_velocity else 'n/a'}",
        "Doppler robustness is recorded as reconstruction error normalized by temporal channel variation.",
        "",
        "## Required Questions",
        f"1. PICASSO vs DnCNN under CDL: {_beats(means, 'PICASSO-rec', 'Enhanced-DnCNN')}.",
        f"2. Physics loss under mobility: {_beats(means, 'PICASSO-rec-physics', 'PICASSO-rec')}.",
        "3. GAN usefulness: not supported in Stage 4 main experiments; GAN remains disabled by policy after Stage 3B showed weak or negative gains.",
        "4. Doppler sensitivity: compare velocity rows in stage4_summary.csv; higher velocity generally raises the Doppler-normalized error.",
        "5. Condition-aware modeling: not promoted here; velocity is recorded as an input condition for Stage 5 but main Stage 4 models intentionally avoid increasing architecture.",
        "6. Synthetic-stage conclusion: stability of supervised PICASSO remains the key question under CDL; GAN remains unnecessary unless later realistic data contradicts this result.",
        "",
        "## Stage 5 Recommendation",
        "Move toward paper writing only after a slightly larger CDL run confirms the same ordering. If PICASSO-rec remains competitive, Stage 5 should emphasize supervised physics-guided reconstruction under realistic CDL mobility rather than GAN synthesis.",
    ])
    return "\n".join(lines) + "\n"


def _beats(means: dict[str, float], lhs: str, rhs: str) -> str:
    if lhs not in means or rhs not in means:
        return "not available"
    return f"{'yes' if means[lhs] < means[rhs] else 'no'} ({lhs}={means[lhs]:.8f}, {rhs}={means[rhs]:.8f})"


def _means(rows: list[dict[str, object]]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["nmse"]))
    return {method: mean(values) for method, values in grouped.items()}


def _filtered_means(rows: list[dict[str, object]], predicate) -> dict[str, float]:
    return _means([row for row in rows if predicate(row)])


def _row(config: dict[str, Any], method: str, profile: str, pattern: str, seed: int, pilot_ratio: float, snr_db: float, velocity: float, metrics: dict[str, float], runtime: float) -> dict[str, object]:
    return {
        "stage": config["project"]["stage"],
        "method": method,
        "profile": profile,
        "pilot_pattern": pattern,
        "seed": seed,
        "pilot_ratio": f"{pilot_ratio:.4f}",
        "snr_db": f"{snr_db:g}",
        "velocity_kmh": f"{velocity:g}",
        "nmse": f"{metrics['nmse']:.8f}",
        "mse": f"{metrics['mse']:.8f}",
        "mae": f"{metrics['mae']:.8f}",
        "pilot_consistency_error": f"{metrics['pilot_consistency_error']:.8f}",
        "delay_domain_sparsity_score": f"{metrics['delay_domain_sparsity_score']:.8f}",
        "doppler_robustness_metric": f"{metrics['doppler_robustness_metric']:.8f}",
        "runtime_seconds": f"{runtime:.4f}",
    }


def _clip(model: torch.nn.Module, config: dict[str, Any]) -> None:
    max_norm = float(config["training"].get("gradient_clip_norm", 0.0))
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 4 must not save generated data.")
    if bool(config["artifact_policy"].get("save_checkpoints", False)):
        raise ValueError("Stage 4 main run must not save checkpoints.")


def _resolve_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested.lower() == "cuda" and torch.cuda.is_available() else "cpu")


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
        raise ValueError(f"Config {path!r} must contain YAML mapping.")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 CDL generalization diagnostics.")
    parser.add_argument("--config", default="configs/stage4_cdl.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
