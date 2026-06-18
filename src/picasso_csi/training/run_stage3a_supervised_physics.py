"""Run Stage 3A supervised physics-guided reconstruction experiments."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
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
    STAGE3A_FIELDS,
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


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_stage3a(config)
    print(f"device: {result['device']}")
    print(f"grid type: {result['scope']}")
    print(f"total rows: {result['rows']}")
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"best method overall: {result['best_method_overall']}")
    print(f"best method under low pilot ratio: {result['best_method_low_pilot']}")
    print(f"best method under low SNR: {result['best_method_low_snr']}")
    print(f"PICASSO-rec beats Enhanced-DnCNN: {result['picasso_rec_beats_enhanced']}")
    print(f"PICASSO-rec-physics beats Enhanced-DnCNN: {result['picasso_physics_beats_enhanced']}")
    print(f"PICASSO-full-light-adv beats PICASSO-rec-physics: {result['light_adv_beats_physics']}")
    print("output files:")
    for path in result["output_files"]:
        print(f"  {path}")


def run_stage3a(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    scope = _select_scope(config)
    device = _resolve_device(str(config["training"].get("device", "cuda")))
    table = ResultTable()
    methods = _method_specs(config)

    print(f"device: {device}")
    print(f"grid type: {scope['name']}")
    print(f"total combinations: {len(scope['seeds']) * len(scope['pilot_ratios']) * len(scope['snr_db_values'])}")
    print(f"methods: {[method['name'] for method in methods]}")
    print(f"seeds: {scope['seeds']}")
    print(f"pilot ratios: {scope['pilot_ratios']}")
    print(f"SNR values: {scope['snr_db_values']}")

    for seed in scope["seeds"]:
        _set_seed(int(seed))
        for pilot_ratio in scope["pilot_ratios"]:
            for snr_db in scope["snr_db_values"]:
                print(f"running seed={seed}, pilot_ratio={pilot_ratio}, snr_db={snr_db}")
                train_loader, val_loader, test_loader = _build_loaders(
                    config, scope, int(seed), float(pilot_ratio), float(snr_db)
                )
                for spec in methods:
                    method_start = time.perf_counter()
                    print(f"  current method: {spec['name']}")
                    metrics, epochs_used, val_nmse = _run_method(
                        spec, train_loader, val_loader, test_loader, device, config, scope
                    )
                    runtime_seconds = time.perf_counter() - method_start
                    table.append(
                        _make_row(
                            config=config,
                            scope=scope,
                            spec=spec,
                            seed=int(seed),
                            pilot_ratio=float(pilot_ratio),
                            snr_db=float(snr_db),
                            metrics=metrics,
                            epochs_used=epochs_used,
                            runtime_seconds=runtime_seconds,
                        )
                    )
                    print(f"    validation NMSE: {val_nmse:.6f}")
                    print(f"    test NMSE: {metrics['nmse']:.6f}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    runtime_seconds = time.perf_counter() - start
    output_files = _save_outputs(config, table, scope, runtime_seconds)
    summary = table.grouped_summary()
    print("mean NMSE by method:")
    for row in summary:
        print(f"  {row['method']}: {row['nmse_mean']}")

    result = _summarize_result(table)
    result.update(
        {
            "device": str(device),
            "scope": scope["name"],
            "rows": len(table.rows),
            "runtime_seconds": runtime_seconds,
            "output_files": output_files,
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
) -> tuple[dict[str, float], int, float]:
    if spec["kind"] == "ls":
        val_metrics = _evaluate_estimator(val_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))
        test_metrics = _evaluate_estimator(test_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))
        return test_metrics, 0, val_metrics["nmse"]

    if spec["kind"] == "dncnn":
        model = (EnhancedDnCNNBaseline() if spec["enhanced"] else DnCNNBaseline()).to(device)
        epochs_used, val_nmse = _train_supervised_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=int(scope["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            patience=int(scope["early_stop_patience"]),
            estimator=lambda batch: model(batch["H_sparse"], batch["mask"]),
        )
        metrics = _evaluate_estimator(test_loader, device, lambda batch: model(batch["H_sparse"], batch["mask"]))
        return metrics, epochs_used, val_nmse

    generator = PICASSOGenerator(base_channels=64, num_blocks=4).to(device)
    discriminator = PICASSODiscriminator(base_channels=32).to(device) if spec["uses_adv"] else None
    epochs_used, val_nmse = _train_picasso_model(
        generator,
        discriminator,
        train_loader,
        val_loader,
        device,
        config,
        scope,
        loss_mode=str(spec["loss_mode"]),
    )
    metrics = _evaluate_estimator(test_loader, device, lambda batch: generator(batch["H_sparse"]))
    return metrics, epochs_used, val_nmse


def _train_supervised_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> tuple[int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_val = float("inf")
    stale_epochs = 0
    epochs_used = 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(estimator(batch), batch["H_full"])
            loss.backward()
            optimizer.step()
        epochs_used = epoch + 1
        val_nmse = _evaluate_estimator(val_loader, device, estimator)["nmse"]
        if val_nmse + 1e-7 < best_val:
            best_val = val_nmse
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    return epochs_used, best_val


def _train_picasso_model(
    generator: PICASSOGenerator,
    discriminator: PICASSODiscriminator | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    scope: dict[str, Any],
    loss_mode: str,
) -> tuple[int, float]:
    loss_config = config["loss"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["learning_rate"]))
    optimizer_d = (
        torch.optim.Adam(discriminator.parameters(), lr=float(config["training"]["learning_rate"]))
        if discriminator is not None
        else None
    )
    best_val = float("inf")
    stale_epochs = 0
    epochs_used = 0
    for epoch in range(int(scope["epochs"])):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            if discriminator is not None and optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    H_fake = generator(batch["H_sparse"])
                d_loss = discriminator_bce_loss(discriminator(batch["H_full"]), discriminator(H_fake.detach()))
                d_loss.backward()
                optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            H_hat = generator(batch["H_sparse"])
            fake_logits = discriminator(H_hat) if discriminator is not None else None
            losses = picasso_generator_loss(
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
                loss_mode=loss_mode,
            )
            losses["total"].backward()
            optimizer_g.step()
        epochs_used = epoch + 1
        val_nmse = _evaluate_estimator(val_loader, device, lambda batch: generator(batch["H_sparse"]))["nmse"]
        if val_nmse + 1e-7 < best_val:
            best_val = val_nmse
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(scope["early_stop_patience"]):
            break
    return epochs_used, best_val


def _build_loaders(
    config: dict[str, Any],
    scope: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = int(config["data"]["batch_size"])
    train_dataset = _build_dataset(config, int(scope["train_samples"]), seed, pilot_ratio, snr_db)
    val_dataset = _build_dataset(config, int(scope["val_samples"]), seed + 50_000, pilot_ratio, snr_db)
    test_dataset = _build_dataset(config, int(scope["test_samples"]), seed + 100_000, pilot_ratio, snr_db)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _build_dataset(
    config: dict[str, Any],
    num_samples: int,
    seed: int,
    pilot_ratio: float,
    snr_db: float,
) -> NoisySyntheticCSIDataset:
    system = config["system"]
    channel = config.get("channel", {})
    return NoisySyntheticCSIDataset(
        num_samples=num_samples,
        n_tx=int(system["n_tx"]),
        n_rx=int(system["n_rx"]),
        n_subcarriers=int(system["n_subcarriers"]),
        n_paths=int(system["n_paths"]),
        pilot_ratio=pilot_ratio,
        snr_db=snr_db,
        seed=seed,
        random_n_paths=bool(channel.get("random_n_paths", False)),
        min_paths=int(channel.get("min_paths", 3)),
        max_paths=int(channel.get("max_paths", 10)),
        delay_spread=channel.get("delay_spread", 1.0),
        gain_distribution=str(channel.get("gain_distribution", "complex_gaussian")),
        normalize_channel=bool(channel.get("normalize_channel", False)),
        pilot_noise_only=bool(channel.get("pilot_noise_only", True)),
    )


@torch.no_grad()
def _evaluate_estimator(
    loader: DataLoader,
    device: torch.device,
    estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor],
) -> dict[str, float]:
    totals = {key: 0.0 for key in ["samples", "nmse", "mse", "mae", "pilot_consistency_error", "delay_sparsity_score"]}
    for batch in loader:
        batch = _to_device(batch, device)
        H_hat = estimator(batch)
        batch_size = float(batch["H_full"].shape[0])
        totals["samples"] += batch_size
        totals["nmse"] += float(nmse(H_hat, batch["H_full"]).detach().cpu()) * batch_size
        totals["mse"] += float(mse(H_hat, batch["H_full"]).detach().cpu()) * batch_size
        totals["mae"] += float(mae(H_hat, batch["H_full"]).detach().cpu()) * batch_size
        totals["pilot_consistency_error"] += float(
            pilot_consistency_error(H_hat, batch["H_sparse"], batch["mask"]).detach().cpu()
        ) * batch_size
        totals["delay_sparsity_score"] += float(delay_domain_sparsity_score(H_hat).detach().cpu()) * batch_size
    samples = max(totals["samples"], 1.0)
    return {key: value / samples for key, value in totals.items() if key != "samples"}


def _method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config["models"]
    specs: list[dict[str, Any]] = []
    if models.get("run_ls", True):
        specs.append(_spec("LS", "ls"))
    if models.get("run_dncnn", True):
        specs.append(_spec("DnCNN", "dncnn"))
    if models.get("run_enhanced_dncnn", True):
        specs.append(_spec("Enhanced-DnCNN", "dncnn", enhanced=True))
    if models.get("run_picasso_rec", True):
        specs.append(_spec("PICASSO-rec", "picasso", loss_mode="rec_only"))
    if models.get("run_picasso_rec_physics", True):
        specs.append(_spec("PICASSO-rec-physics", "picasso", loss_mode="rec_physics"))
    if models.get("run_picasso_full_light_adv", True):
        specs.append(_spec("PICASSO-full-light-adv", "picasso", loss_mode="full", uses_adv=True))
    return specs


def _spec(
    name: str,
    kind: str,
    loss_mode: str = "",
    uses_adv: bool = False,
    enhanced: bool = False,
) -> dict[str, Any]:
    return {"name": name, "kind": kind, "loss_mode": loss_mode, "uses_adv": uses_adv, "enhanced": enhanced}


def _select_scope(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    training = config["training"]
    diagnosis = config.get("diagnosis", {})
    full_combo_count = len(data["seeds"]) * len(data["pilot_ratios"]) * len(data["snr_db_values"])
    estimated_units = full_combo_count * len(_method_specs(config)) * int(training["epochs"]) * int(data["train_samples"])
    too_large = estimated_units > 30_000_000
    use_reduced = bool(training.get("allow_reduced_grid_if_runtime_high", True)) and too_large
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
            "early_stop_patience": int(training["early_stop_patience"]),
        }
    return {
        "name": "reduced",
        "reason": f"full grid estimated units {estimated_units} exceeded the diagnostic budget",
        "seeds": list(diagnosis.get("reduced_grid_seeds", data["seeds"])),
        "pilot_ratios": list(diagnosis.get("reduced_grid_pilot_ratios", data["pilot_ratios"])),
        "snr_db_values": list(diagnosis.get("reduced_grid_snr_db_values", data["snr_db_values"])),
        "train_samples": int(diagnosis.get("reduced_train_samples", min(int(data["train_samples"]), 1024))),
        "val_samples": int(diagnosis.get("reduced_val_samples", min(int(data["val_samples"]), 256))),
        "test_samples": int(diagnosis.get("reduced_test_samples", min(int(data["test_samples"]), 512))),
        "epochs": int(diagnosis.get("reduced_epochs", min(int(training["epochs"]), 5))),
        "early_stop_patience": int(diagnosis.get("reduced_early_stop_patience", min(int(training["early_stop_patience"]), 3))),
    }


def _save_outputs(config: dict[str, Any], table: ResultTable, scope: dict[str, Any], runtime_seconds: float) -> list[str]:
    outputs = config["outputs"]
    results_dir = Path(outputs["results_dir"])
    raw_path = table.save_raw_csv_with_fields(results_dir / outputs["raw_csv"], STAGE3A_FIELDS)
    summary_path = table.save_summary_by_condition_csv(results_dir / outputs["summary_csv"])
    paper_path = table.save_paper_table_csv(results_dir / outputs["paper_table_csv"])
    diagnosis_path = table.save_diagnosis_csv(results_dir / outputs["diagnosis_csv"])
    analysis_path = _write_analysis(config, table, scope, runtime_seconds)
    return [str(path) for path in [raw_path, summary_path, paper_path, diagnosis_path, analysis_path]]


def _write_analysis(config: dict[str, Any], table: ResultTable, scope: dict[str, Any], runtime_seconds: float) -> Path:
    path = Path(config["outputs"]["analysis_md"])
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = table.grouped_summary()
    result = _summarize_result(table)
    lines = [
        "# Stage 3A Supervised Physics-Guided Reconstruction",
        "",
        "## Stage 3A Goal",
        "Stage 3A tests supervised physics-guided CSI reconstruction before committing to long full-GAN training.",
        "",
        "## Why Full GAN Is Deferred",
        "Stage 2BC showed that adversarial variants can help in a reduced diagnostic, but the evidence is not yet strong enough for long GAN training. The key gate is whether supervised PICASSO-rec-physics can beat Enhanced-DnCNN.",
        "",
        "## Why Supervised Physics First",
        "A supervised reconstruction model is easier to train, easier to diagnose, and directly tests whether pilot consistency, frequency smoothness, and delay sparsity add value beyond a stronger CNN baseline.",
        "",
        "## Experiment Setup",
        f"- Grid type: {scope['name']}",
        f"- Reason: {scope['reason']}",
        f"- Runtime seconds: {runtime_seconds:.2f}",
        f"- Seeds: {scope['seeds']}",
        f"- Pilot ratios: {scope['pilot_ratios']}",
        f"- SNR values: {scope['snr_db_values']}",
        f"- Train/val/test samples: {scope['train_samples']} / {scope['val_samples']} / {scope['test_samples']}",
        f"- Epochs and early-stop patience: {scope['epochs']} / {scope['early_stop_patience']}",
        "",
        "## Dataset Difficulty",
        "The diagnostic uses random path counts, random delay spread, complex Gaussian path gains, normalized clean channels, and pilot-only AWGN.",
        "",
        "## Model Comparison",
        "Methods: LS, DnCNN, Enhanced-DnCNN, PICASSO-rec, PICASSO-rec-physics, PICASSO-full-light-adv.",
        "",
        "## Loss Setup",
        f"- lambda_adv: {config['loss']['lambda_adv']}",
        f"- lambda_pilot: {config['loss']['lambda_pilot']}",
        f"- lambda_smooth: {config['loss']['lambda_smooth']}",
        f"- lambda_sparse: {config['loss']['lambda_sparse']}",
        "",
        "## Results Overview",
    ]
    for row in summary:
        lines.append(f"- {row['method']}: NMSE {row['nmse_mean']} +/- {row['nmse_std']} over {row['count']} rows")
    lines.extend(
        [
            "",
            "## Low Pilot Ratio Analysis",
            f"Best method for pilot_ratio <= 0.125: {result['best_method_low_pilot']}.",
            "",
            "## Low SNR Analysis",
            f"Best method for SNR <= 10 dB: {result['best_method_low_snr']}.",
            "",
            "## PICASSO-rec-physics vs Enhanced-DnCNN",
            f"PICASSO-rec-physics beats Enhanced-DnCNN: {result['picasso_physics_beats_enhanced']}.",
            "",
            "## PICASSO-full-light-adv vs PICASSO-rec-physics",
            f"PICASSO-full-light-adv beats PICASSO-rec-physics: {result['light_adv_beats_physics']}.",
            "",
            "## Stage 3B Recommendation",
            _stage3b_recommendation(result),
            "",
            "## 3GPP CDL / QuaDRiGA Recommendation",
            "Introduce 3GPP CDL or QuaDRiGa before claiming paper-level realism. The synthetic generator is useful for method gating but not sufficient as final channel evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stage3b_recommendation(result: dict[str, object]) -> str:
    if result["picasso_physics_beats_enhanced"] and result["light_adv_beats_physics"]:
        return "Proceed to Stage 3B with cautious light-adversarial training and keep supervised physics as the main baseline."
    if result["picasso_physics_beats_enhanced"]:
        return "Proceed to Stage 3B by scaling supervised physics-guided reconstruction first; keep GAN as a secondary ablation."
    return "Pause long GAN training. Improve supervised physics-guided reconstruction and channel realism before Stage 3B."


def _summarize_result(table: ResultTable) -> dict[str, object]:
    means = {row["method"]: float(row["nmse_mean"]) for row in table.grouped_summary()}
    return {
        "best_method_overall": min(means, key=means.get) if means else "",
        "best_method_low_pilot": table.winner(lambda row: float(row["pilot_ratio"]) <= 0.125),
        "best_method_low_snr": table.winner(lambda row: float(row["snr_db"]) <= 10),
        "picasso_rec_beats_enhanced": bool(
            "PICASSO-rec" in means and "Enhanced-DnCNN" in means and means["PICASSO-rec"] < means["Enhanced-DnCNN"]
        ),
        "picasso_physics_beats_enhanced": bool(
            "PICASSO-rec-physics" in means
            and "Enhanced-DnCNN" in means
            and means["PICASSO-rec-physics"] < means["Enhanced-DnCNN"]
        ),
        "light_adv_beats_physics": bool(
            "PICASSO-full-light-adv" in means
            and "PICASSO-rec-physics" in means
            and means["PICASSO-full-light-adv"] < means["PICASSO-rec-physics"]
        ),
    }


def _make_row(
    config: dict[str, Any],
    scope: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
    metrics: dict[str, float],
    epochs_used: int,
    runtime_seconds: float,
) -> dict[str, object]:
    loss = config["loss"]
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
        "lambda_pilot": loss["lambda_pilot"] if "physics" in spec["name"] or spec["uses_adv"] else 0.0,
        "lambda_smooth": loss["lambda_smooth"] if "physics" in spec["name"] or spec["uses_adv"] else 0.0,
        "lambda_sparse": loss["lambda_sparse"] if "physics" in spec["name"] or spec["uses_adv"] else 0.0,
    }


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 3A must not save generated datasets.")
    policy = config.get("artifact_policy", {})
    blocked = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Stage 3A must not save protected artifacts: {enabled}.")
    if not bool(policy.get("save_csv_results", False)):
        raise ValueError("Stage 3A requires CSV metrics output.")


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
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 3A supervised physics diagnostics.")
    parser.add_argument("--config", default="configs/stage3a_supervised_physics.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
