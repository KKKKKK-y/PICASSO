"""Run Stage 2B-2C comprehensive diagnostic experiments."""

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
PROJECT_ROOT = SRC_ROOT.parents[0]
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
from picasso_csi.models import (  # noqa: E402
    DnCNNBaseline,
    EnhancedDnCNNBaseline,
    PICASSODiscriminator,
    PICASSOGenerator,
    ls_baseline,
)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_stage2bc(config)
    print(f"device: {result['device']}")
    print(f"scope: {result['scope']}")
    print(f"total rows: {result['rows']}")
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"best method overall: {result['best_method_overall']}")
    print(f"best method under low pilot: {result['best_method_low_pilot']}")
    print(f"best method under low SNR: {result['best_method_low_snr']}")
    print(f"PICASSO beats LS: {result['picasso_beats_ls']}")
    print(f"PICASSO beats DnCNN: {result['picasso_beats_dncnn']}")
    print("output files:")
    for path in result["output_files"]:
        print(f"  {path}")


def run_stage2bc(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    scope = _select_scope(config)
    device = _resolve_device(str(config["training"].get("device", "cuda")))
    table = ResultTable()
    methods = _method_specs(config)

    print(f"device: {device}")
    print(f"full grid or reduced grid: {scope['name']}")
    print(f"total combinations: {len(scope['seeds']) * len(scope['pilot_ratios']) * len(scope['snr_db_values'])}")
    print(f"methods: {[spec['name'] for spec in methods]}")
    print(f"seeds: {scope['seeds']}")
    print(f"pilot ratios: {scope['pilot_ratios']}")
    print(f"SNR values: {scope['snr_db_values']}")

    for seed in scope["seeds"]:
        _set_seed(int(seed))
        for pilot_ratio in scope["pilot_ratios"]:
            for snr_db in scope["snr_db_values"]:
                print(f"running seed={seed}, pilot_ratio={pilot_ratio}, snr_db={snr_db}")
                train_loader, test_loader = _build_loaders(config, scope, int(seed), float(pilot_ratio), float(snr_db))
                for spec in methods:
                    method_start = time.perf_counter()
                    metrics = _run_method(spec, train_loader, test_loader, device, config, scope)
                    runtime_seconds = time.perf_counter() - method_start
                    row = _make_row(
                        config=config,
                        scope=scope,
                        spec=spec,
                        seed=int(seed),
                        pilot_ratio=float(pilot_ratio),
                        snr_db=float(snr_db),
                        metrics=metrics,
                        runtime_seconds=runtime_seconds,
                    )
                    table.append(row)
                    print(f"  {spec['name']} NMSE: {metrics['nmse']:.6f} ({runtime_seconds:.1f}s)")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    output_files = _save_outputs(config, table, scope, time.perf_counter() - start)
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
            "runtime_seconds": time.perf_counter() - start,
            "output_files": output_files,
        }
    )
    return result


def _run_method(
    spec: dict[str, Any],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, float]:
    if spec["kind"] == "ls":
        return _evaluate_estimator(test_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))

    if spec["kind"] == "dncnn":
        model_cls = EnhancedDnCNNBaseline if spec["enhanced"] else DnCNNBaseline
        model = model_cls(use_condition=spec["use_condition"]).to(device)
        _train_supervised(
            model,
            train_loader,
            device,
            epochs=scope["epochs_main"],
            learning_rate=float(config["training"]["learning_rate_dncnn"]),
            use_condition=spec["use_condition"],
        )
        return _evaluate_estimator(test_loader, device, lambda batch: _model_forward(model, batch, spec["use_condition"]))

    generator = PICASSOGenerator(base_channels=64, num_blocks=4, use_condition=spec["use_condition"]).to(device)
    discriminator = PICASSODiscriminator(base_channels=32).to(device) if spec["uses_adv"] else None
    _train_picasso(
        generator,
        discriminator,
        train_loader,
        device,
        config,
        epochs=scope["epochs_diagnostic"] if spec["loss_mode"] != "full" else scope["epochs_main"],
        loss_mode=spec["loss_mode"],
        use_condition=spec["use_condition"],
    )
    return _evaluate_estimator(test_loader, device, lambda batch: _model_forward(generator, batch, spec["use_condition"]))


def _build_loaders(
    config: dict[str, Any],
    scope: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = _build_dataset(config, scope["train_samples"], seed, pilot_ratio, snr_db)
    test_dataset = _build_dataset(config, scope["test_samples"], seed + 100_000, pilot_ratio, snr_db)
    batch_size = int(config["data"]["batch_size"])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
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


def _train_supervised(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    use_condition: bool,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            H_hat = _model_forward(model, batch, use_condition)
            loss = loss_fn(H_hat, batch["H_full"])
            loss.backward()
            optimizer.step()


def _train_picasso(
    generator: PICASSOGenerator,
    discriminator: PICASSODiscriminator | None,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    epochs: int,
    loss_mode: str,
    use_condition: bool,
) -> None:
    loss_config = config["loss"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["learning_rate_g"]))
    optimizer_d = (
        torch.optim.Adam(discriminator.parameters(), lr=float(config["training"]["learning_rate_d"]))
        if discriminator is not None
        else None
    )
    for _ in range(epochs):
        generator.train()
        if discriminator is not None:
            discriminator.train()
        for batch in loader:
            batch = _to_device(batch, device)
            if discriminator is not None and optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    H_fake = _model_forward(generator, batch, use_condition)
                d_loss = discriminator_bce_loss(discriminator(batch["H_full"]), discriminator(H_fake.detach()))
                d_loss.backward()
                optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            H_hat = _model_forward(generator, batch, use_condition)
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


def _model_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor], use_condition: bool) -> torch.Tensor:
    if isinstance(model, PICASSOGenerator):
        if use_condition:
            return model(batch["H_sparse"], batch["snr_db"], batch["pilot_ratio"])
        return model(batch["H_sparse"])
    if use_condition:
        return model(batch["H_sparse"], batch["mask"], batch["snr_db"], batch["pilot_ratio"])
    return model(batch["H_sparse"], batch["mask"])


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config["models"]
    specs: list[dict[str, Any]] = []
    if models.get("run_ls", True):
        specs.append(_spec("LS", "ls"))
    if models.get("run_dncnn", True):
        specs.append(_spec("DnCNN", "dncnn"))
    if models.get("run_cond_dncnn", True):
        specs.append(_spec("Cond-DnCNN", "dncnn", use_condition=True))
    if models.get("run_enhanced_dncnn", True):
        specs.append(_spec("Enhanced-DnCNN", "dncnn", enhanced=True))
    if models.get("run_picasso_rec", True):
        specs.append(_spec("PICASSO-rec", "picasso", loss_mode="rec_only"))
    if models.get("run_picasso_rec_physics", True):
        specs.append(_spec("PICASSO-rec-physics", "picasso", loss_mode="rec_physics"))
    if models.get("run_picasso_rec_adv", True):
        specs.append(_spec("PICASSO-rec-adv", "picasso", loss_mode="rec_adv", uses_adv=True))
    if models.get("run_picasso_full", True):
        specs.append(_spec("PICASSO-full", "picasso", loss_mode="full", uses_adv=True))
    if models.get("run_picasso_cond_full", True):
        specs.append(_spec("PICASSO-cond-full", "picasso", loss_mode="full", uses_adv=True, use_condition=True))
    return specs


def _spec(
    name: str,
    kind: str,
    loss_mode: str = "",
    uses_adv: bool = False,
    use_condition: bool = False,
    enhanced: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "loss_mode": loss_mode,
        "uses_adv": uses_adv,
        "use_condition": use_condition,
        "enhanced": enhanced,
    }


def _select_scope(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    training = config["training"]
    diagnosis = config.get("diagnosis", {})
    full_combo_count = len(data["seeds"]) * len(data["pilot_ratios"]) * len(data["snr_db_values"])
    estimated_units = full_combo_count * len(_method_specs(config)) * int(training["epochs_main"]) * int(data["train_samples"])
    too_large = estimated_units > 15_000_000
    use_reduced = bool(diagnosis.get("allow_reduced_grid_if_runtime_high", True)) and too_large
    if not use_reduced:
        return {
            "name": "full",
            "reason": "full grid estimate is within the diagnostic budget",
            "seeds": list(data["seeds"]),
            "pilot_ratios": list(data["pilot_ratios"]),
            "snr_db_values": list(data["snr_db_values"]),
            "train_samples": int(data["train_samples"]),
            "test_samples": int(data["test_samples"]),
            "epochs_main": int(training["epochs_main"]),
            "epochs_diagnostic": int(training["epochs_diagnostic"]),
        }
    return {
        "name": "reduced",
        "reason": f"full grid estimated units {estimated_units} exceeded the 5-hour diagnostic budget",
        "seeds": list(diagnosis.get("reduced_grid_seeds", data["seeds"][:1])),
        "pilot_ratios": list(diagnosis.get("reduced_grid_pilot_ratios", data["pilot_ratios"])),
        "snr_db_values": list(diagnosis.get("reduced_grid_snr_db_values", data["snr_db_values"])),
        "train_samples": int(diagnosis.get("reduced_train_samples", min(int(data["train_samples"]), 512))),
        "test_samples": int(diagnosis.get("reduced_test_samples", min(int(data["test_samples"]), 256))),
        "epochs_main": int(diagnosis.get("reduced_epochs_main", min(int(training["epochs_main"]), 3))),
        "epochs_diagnostic": int(diagnosis.get("reduced_epochs_diagnostic", min(int(training["epochs_diagnostic"]), 2))),
    }


def _save_outputs(config: dict[str, Any], table: ResultTable, scope: dict[str, Any], runtime_seconds: float) -> list[str]:
    outputs = config["outputs"]
    results_dir = Path(outputs["results_dir"])
    raw_path = table.save_raw_csv(results_dir / outputs["raw_csv"])
    summary_path = table.save_summary_csv(results_dir / outputs["summary_csv"])
    paper_path = table.save_paper_table_csv(results_dir / outputs["paper_table_csv"])
    diagnosis_path = table.save_diagnosis_csv(results_dir / outputs["diagnosis_csv"])
    ablation_path = write_csv(_ablation_rows(table), results_dir / outputs["ablation_csv"], ["comparison", "winner", "details"])
    by_pilot_path = write_csv(_group_metric_rows(table, "pilot_ratio"), results_dir / outputs["nmse_by_pilot_csv"], ["pilot_ratio", "method", "nmse_mean", "count"])
    by_snr_path = write_csv(_group_metric_rows(table, "snr_db"), results_dir / outputs["nmse_by_snr_csv"], ["snr_db", "method", "nmse_mean", "count"])
    analysis_path = _write_analysis(config, table, scope, runtime_seconds)
    return [str(path) for path in [raw_path, summary_path, paper_path, diagnosis_path, ablation_path, by_pilot_path, by_snr_path, analysis_path]]


def _ablation_rows(table: ResultTable) -> list[dict[str, object]]:
    return [
        {"comparison": "condition_dncnn", "winner": _winner_between(table, "DnCNN", "Cond-DnCNN"), "details": "DnCNN vs Cond-DnCNN overall NMSE"},
        {"comparison": "enhanced_dncnn", "winner": _winner_between(table, "DnCNN", "Enhanced-DnCNN"), "details": "DnCNN vs Enhanced-DnCNN overall NMSE"},
        {"comparison": "physics_loss", "winner": _winner_between(table, "PICASSO-rec", "PICASSO-rec-physics"), "details": "Supervised PICASSO with vs without physics loss"},
        {"comparison": "adversarial_loss", "winner": _winner_between(table, "PICASSO-rec", "PICASSO-rec-adv"), "details": "Supervised PICASSO with vs without adversarial loss"},
        {"comparison": "condition_picasso", "winner": _winner_between(table, "PICASSO-full", "PICASSO-cond-full"), "details": "PICASSO full vs condition-aware full"},
    ]


def _group_metric_rows(table: ResultTable, key: str) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object], list[float]] = {}
    for row in table.rows:
        grouped.setdefault((row[key], row["method"]), []).append(float(row["nmse"]))
    return [
        {key: group_key, "method": method, "nmse_mean": f"{np.mean(values):.8f}", "count": len(values)}
        for (group_key, method), values in sorted(grouped.items(), key=lambda item: (float(item[0][0]), str(item[0][1])))
    ]


def _write_analysis(config: dict[str, Any], table: ResultTable, scope: dict[str, Any], runtime_seconds: float) -> Path:
    path = Path(config["outputs"]["analysis_md"])
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = table.grouped_summary()
    best = _summarize_result(table)
    lines = [
        "# Stage 2BC Results Analysis",
        "",
        "## Experiment Scope",
        f"- Scope: {scope['name']}",
        f"- Reason: {scope['reason']}",
        f"- Runtime seconds: {runtime_seconds:.2f}",
        f"- Seeds: {scope['seeds']}",
        f"- Pilot ratios: {scope['pilot_ratios']}",
        f"- SNR values: {scope['snr_db_values']}",
        f"- Train/test samples: {scope['train_samples']} / {scope['test_samples']}",
        f"- Epochs main/diagnostic: {scope['epochs_main']} / {scope['epochs_diagnostic']}",
        "",
        "## Dataset Difficulty",
        "- The dataset uses random path counts, random delay spread, complex Gaussian gains, normalized clean channels, and pilot-only AWGN.",
        "- Labels remain clean H_full tensors, while H_sparse stores noisy pilot observations and zeros elsewhere.",
        "",
        "## Overall Results",
    ]
    for row in summary:
        lines.append(f"- {row['method']}: NMSE {row['nmse_mean']} +/- {row['nmse_std']} over {row['count']} rows")
    lines.extend(
        [
            "",
            "## Low Pilot Ratio Results",
            f"- Best method for pilot_ratio <= 0.125: {best['best_method_low_pilot']}",
            "",
            "## Low SNR Results",
            f"- Best method for SNR <= 10 dB: {best['best_method_low_snr']}",
            "",
            "## Condition-Aware Effect",
            f"- DnCNN vs Cond-DnCNN winner: {_winner_between(table, 'DnCNN', 'Cond-DnCNN')}",
            f"- PICASSO-full vs PICASSO-cond-full winner: {_winner_between(table, 'PICASSO-full', 'PICASSO-cond-full')}",
            "",
            "## Physics Loss Effect",
            f"- PICASSO-rec vs PICASSO-rec-physics winner: {_winner_between(table, 'PICASSO-rec', 'PICASSO-rec-physics')}",
            "",
            "## Adversarial Loss Effect",
            f"- PICASSO-rec vs PICASSO-rec-adv winner: {_winner_between(table, 'PICASSO-rec', 'PICASSO-rec-adv')}",
            f"- PICASSO-rec-physics vs PICASSO-full winner: {_winner_between(table, 'PICASSO-rec-physics', 'PICASSO-full')}",
            "",
            "## Does PICASSO Beat Baselines?",
            f"- PICASSO beats LS: {best['picasso_beats_ls']}",
            f"- PICASSO beats DnCNN: {best['picasso_beats_dncnn']}",
            f"- PICASSO beats Enhanced-DnCNN: {best['picasso_beats_enhanced_dncnn']}",
            "- These answers are based on the best overall PICASSO-family mean NMSE versus the listed baseline mean NMSE.",
            "",
            "## Honest Diagnosis",
            "- If the GAN variants do not lead the table, the current evidence favors supervised reconstruction first.",
            "- Likely bottlenecks are task simplicity, zero-filled LS strength at observed pilots, short diagnostic training, and adversarial instability under small budgets.",
            "- Physics losses can help only if their weights match the channel generator; otherwise pilot and sparsity penalties may over-constrain reconstruction.",
            "",
            "## Stage 3 Recommendation",
            "- Do not enter long formal GAN training until a supervised or physics-guided variant clearly beats Enhanced-DnCNN on the synthetic diagnostic.",
            "- Prioritize supervised physics-guided reconstruction and stronger realistic channel data.",
            "- Add 3GPP CDL or QuaDRiGa channels before using this as an IEEE Communications Letters paper result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summarize_result(table: ResultTable) -> dict[str, object]:
    means = {row["method"]: float(row["nmse_mean"]) for row in table.grouped_summary()}
    picasso_values = {method: value for method, value in means.items() if str(method).startswith("PICASSO")}
    best_picasso = min(picasso_values.values()) if picasso_values else None
    return {
        "best_method_overall": min(means, key=means.get) if means else "",
        "best_method_low_pilot": table.winner(lambda row: float(row["pilot_ratio"]) <= 0.125),
        "best_method_low_snr": table.winner(lambda row: float(row["snr_db"]) <= 10),
        "picasso_beats_ls": bool(best_picasso is not None and "LS" in means and best_picasso < means["LS"]),
        "picasso_beats_dncnn": bool(best_picasso is not None and "DnCNN" in means and best_picasso < means["DnCNN"]),
        "picasso_beats_enhanced_dncnn": bool(
            best_picasso is not None and "Enhanced-DnCNN" in means and best_picasso < means["Enhanced-DnCNN"]
        ),
    }


def _winner_between(table: ResultTable, lhs: str, rhs: str) -> str:
    means = {row["method"]: float(row["nmse_mean"]) for row in table.grouped_summary()}
    if lhs not in means or rhs not in means:
        return "not_available"
    return lhs if means[lhs] <= means[rhs] else rhs


def _make_row(
    config: dict[str, Any],
    scope: dict[str, Any],
    spec: dict[str, Any],
    seed: int,
    pilot_ratio: float,
    snr_db: float,
    metrics: dict[str, float],
    runtime_seconds: float,
) -> dict[str, object]:
    return {
        "stage": config["project"]["stage"],
        "method": spec["name"],
        "loss_mode": spec["loss_mode"],
        "use_condition": spec["use_condition"],
        "pilot_ratio": f"{pilot_ratio:.4f}",
        "snr_db": f"{snr_db:g}",
        "seed": seed,
        "nmse": f"{metrics['nmse']:.8f}",
        "mse": f"{metrics['mse']:.8f}",
        "mae": f"{metrics['mae']:.8f}",
        "pilot_consistency_error": f"{metrics['pilot_consistency_error']:.8f}",
        "delay_sparsity_score": f"{metrics['delay_sparsity_score']:.8f}",
        "epochs": 0 if spec["kind"] == "ls" else scope["epochs_main"],
        "num_train_samples": scope["train_samples"],
        "runtime_seconds": f"{runtime_seconds:.4f}",
    }


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 2BC must not save generated datasets.")
    policy = config.get("artifact_policy", {})
    blocked = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Stage 2BC must not save protected artifacts: {enabled}.")
    if not bool(policy.get("save_csv_results", False)):
        raise ValueError("Stage 2BC requires CSV metrics output.")


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
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 2BC comprehensive diagnostics.")
    parser.add_argument("--config", default="configs/stage2bc_comprehensive.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
