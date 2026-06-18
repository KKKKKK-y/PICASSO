"""Run Stage 3B incremental structural enhancement diagnostics."""

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

from picasso_csi.datasets import NoisySyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import delay_domain_sparsity_score, mae, mse, nmse, pilot_consistency_error, write_csv  # noqa: E402
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
    "loss_mode",
    "use_attention",
    "use_condition",
    "feature_discriminator",
    "runtime_seconds",
]

ABLATION_FIELDS = ["question", "winner", "delta_nmse", "interpretation"]


def main() -> None:
    args = _parse_args()
    result = run_stage3b(_load_config(args.config))
    print(f"device: {result['device']}")
    print(f"rows: {result['rows']}")
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"best method: {result['best_method']}")
    print(f"outputs: {result['outputs']}")


def run_stage3b(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    device = _resolve_device(config["training"]["device"])
    methods = _method_specs(config)
    rows: list[dict[str, object]] = []
    print(f"device: {device}")
    print(f"methods: {[method['name'] for method in methods]}")

    for seed in config["data"]["seeds"]:
        _set_seed(int(seed))
        for pilot_ratio in config["data"]["pilot_ratios"]:
            for snr_db in config["data"]["snr_db_values"]:
                train_loader, test_loader = _build_loaders(config, int(seed), float(pilot_ratio), float(snr_db))
                print(f"seed={seed}, pilot_ratio={pilot_ratio}, snr_db={snr_db}")
                for spec in methods:
                    method_start = time.perf_counter()
                    metrics = _run_method(spec, train_loader, test_loader, device, config)
                    rows.append(_make_row(config, spec, int(seed), float(pilot_ratio), float(snr_db), metrics, time.perf_counter() - method_start))
                    print(f"  {spec['name']} NMSE: {metrics['nmse']:.6f}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    outputs = _save_outputs(config, rows, time.perf_counter() - start)
    means = _method_means(rows)
    return {
        "device": str(device),
        "rows": len(rows),
        "runtime_seconds": time.perf_counter() - start,
        "best_method": min(means, key=means.get),
        "outputs": outputs,
    }


def _run_method(spec: dict[str, Any], train_loader: DataLoader, test_loader: DataLoader, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    if spec["kind"] == "ls":
        return _evaluate(test_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))
    if spec["kind"] == "dncnn":
        model = (EnhancedDnCNNBaseline(hidden_channels=64, depth=8) if spec["enhanced"] else DnCNNBaseline()).to(device)
        _train_supervised(model, train_loader, device, config, lambda batch: model(batch["H_sparse"], batch["mask"]))
        return _evaluate(test_loader, device, lambda batch: model(batch["H_sparse"], batch["mask"]))

    generator = _make_generator(config, spec).to(device)
    discriminator = _make_discriminator(config, spec).to(device) if spec["adv"] else None
    _train_picasso(generator, discriminator, train_loader, device, config, spec)
    return _evaluate(test_loader, device, lambda batch: _gen_forward(generator, batch, spec["condition"]))


def _train_supervised(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: dict[str, Any], estimator: Callable[[dict[str, torch.Tensor]], torch.Tensor]) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    for _ in range(int(config["training"]["epochs"])):
        model.train()
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(estimator(batch), batch["H_full"])
            loss.backward()
            _clip(model, config)
            optimizer.step()


def _train_picasso(generator: PICASSOGenerator, discriminator: PICASSODiscriminator | None, loader: DataLoader, device: torch.device, config: dict[str, Any], spec: dict[str, Any]) -> None:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["learning_rate_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(config["training"]["learning_rate_d"])) if discriminator else None
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        active_mode = "rec_physics" if spec["adv"] and epoch <= int(config["training"]["warmup_epochs"]) else spec["loss_mode"]
        for batch in loader:
            batch = _to_device(batch, device)
            if discriminator and optimizer_d and active_mode == "full":
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = _gen_forward(generator, batch, spec["condition"])
                discriminator_bce_loss(discriminator(batch["H_full"]), discriminator(fake.detach())).backward()
                _clip(discriminator, config)
                optimizer_d.step()
            optimizer_g.zero_grad(set_to_none=True)
            H_hat = _gen_forward(generator, batch, spec["condition"])
            fake_logits = discriminator(H_hat) if discriminator and active_mode == "full" else None
            losses = picasso_generator_loss(
                H_hat,
                batch["H_full"],
                batch["H_sparse"],
                batch["mask"],
                fake_logits=fake_logits,
                lambda_rec=float(config["loss"]["lambda_rec"]),
                lambda_adv=float(config["loss"]["lambda_adv"]),
                lambda_pilot=float(config["loss"]["lambda_pilot"]),
                lambda_smooth=float(config["loss"]["lambda_smooth"]),
                lambda_sparse=float(config["loss"]["lambda_sparse"]),
                lambda_frequency=float(config["loss"]["lambda_frequency"]),
                lambda_energy=float(config["loss"]["lambda_energy"]),
                loss_mode=active_mode,
            )
            losses["total"].backward()
            _clip(generator, config)
            optimizer_g.step()


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
        totals["delay_sparsity_score"] += float(delay_domain_sparsity_score(H_hat).cpu()) * n
    samples = max(totals["samples"], 1.0)
    return {key: totals[key] / samples for key in ["nmse", "mse", "mae", "pilot_consistency_error", "delay_sparsity_score"]}


def _build_loaders(config: dict[str, Any], seed: int, pilot_ratio: float, snr_db: float) -> tuple[DataLoader, DataLoader]:
    batch_size = int(config["data"]["batch_size"])
    train = _dataset(config, int(config["data"]["train_samples"]), seed, pilot_ratio, snr_db)
    test = _dataset(config, int(config["data"]["test_samples"]), seed + 100_000, pilot_ratio, snr_db)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)


def _dataset(config: dict[str, Any], samples: int, seed: int, pilot_ratio: float, snr_db: float) -> NoisySyntheticCSIDataset:
    system = config["system"]
    channel = config["channel"]
    return NoisySyntheticCSIDataset(
        num_samples=samples,
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


def _make_generator(config: dict[str, Any], spec: dict[str, Any]) -> PICASSOGenerator:
    return PICASSOGenerator(
        base_channels=int(config["model_size"]["generator_base_channels"]),
        num_blocks=int(config["model_size"]["generator_num_blocks"]),
        refinement_blocks=int(config["model_size"]["refinement_blocks"]) if spec["enhanced"] else 0,
        use_multiscale_fusion=bool(spec["multiscale"]),
        use_channel_attention=bool(spec["attention"]),
        use_condition=bool(spec["condition"]),
        use_film_conditioning=bool(spec["film"]),
    )


def _make_discriminator(config: dict[str, Any], spec: dict[str, Any]) -> PICASSODiscriminator:
    return PICASSODiscriminator(
        base_channels=int(config["model_size"]["discriminator_base_channels"]),
        use_delay_features=bool(spec["feature_disc"]),
    )


def _gen_forward(generator: PICASSOGenerator, batch: dict[str, torch.Tensor], use_condition: bool) -> torch.Tensor:
    if use_condition:
        return generator(batch["H_sparse"], batch["snr_db"], batch["pilot_ratio"])
    return generator(batch["H_sparse"])


def _method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _spec("LS", "ls"),
        _spec("DnCNN", "dncnn"),
        _spec("Enhanced-DnCNN", "dncnn", enhanced=True),
        _spec("PICASSO-rec-base", "picasso", loss_mode="rec_only"),
        _spec("PICASSO-rec-enhanced", "picasso", loss_mode="rec_only", enhanced=True, multiscale=True),
        _spec("PICASSO-rec-attn", "picasso", loss_mode="rec_only", enhanced=True, multiscale=True, attention=True),
        _spec("PICASSO-rec-physics-enhanced", "picasso", loss_mode="rec_physics", enhanced=True, multiscale=True, attention=True),
        _spec("PICASSO-full-feature", "picasso", loss_mode="full", enhanced=True, multiscale=True, attention=True, adv=True, feature_disc=True),
        _spec("PICASSO-cond-full-feature", "picasso", loss_mode="full", enhanced=True, multiscale=True, attention=True, adv=True, feature_disc=True, condition=True, film=True),
    ]


def _spec(name: str, kind: str, loss_mode: str = "", enhanced: bool = False, multiscale: bool = False, attention: bool = False, adv: bool = False, feature_disc: bool = False, condition: bool = False, film: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "loss_mode": loss_mode,
        "enhanced": enhanced,
        "multiscale": multiscale,
        "attention": attention,
        "adv": adv,
        "feature_disc": feature_disc,
        "condition": condition,
        "film": film,
    }


def _save_outputs(config: dict[str, Any], rows: list[dict[str, object]], runtime_seconds: float) -> list[str]:
    output = config["outputs"]
    result_path = Path(output["results_dir"]) / output["raw_csv"]
    ablation_path = Path(output["results_dir"]) / output["ablation_csv"]
    md_path = Path(output["analysis_md"])
    write_csv(rows, result_path, RAW_FIELDS)
    ablations = _ablation_rows(rows)
    write_csv(ablations, ablation_path, ABLATION_FIELDS)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_analysis_markdown(rows, ablations, runtime_seconds), encoding="utf-8")
    return [str(result_path), str(ablation_path), str(md_path)]


def _ablation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    comparisons = [
        ("architecture", "PICASSO-rec-base", "PICASSO-rec-enhanced", "Does structural enhancement improve PICASSO-rec?"),
        ("attention", "PICASSO-rec-enhanced", "PICASSO-rec-attn", "Does SE attention help?"),
        ("physics", "PICASSO-rec-attn", "PICASSO-rec-physics-enhanced", "Does enhanced physics help?"),
        ("feature_gan", "PICASSO-rec-physics-enhanced", "PICASSO-full-feature", "Does feature-level GAN help?"),
        ("condition", "PICASSO-full-feature", "PICASSO-cond-full-feature", "Does condition-aware FiLM help?"),
    ]
    means = _method_means(rows)
    out = []
    for question, base, variant, text in comparisons:
        if base in means and variant in means:
            delta = means[variant] - means[base]
            winner = variant if delta < 0 else base
            out.append({"question": question, "winner": winner, "delta_nmse": f"{delta:.8f}", "interpretation": text})
    return out


def _analysis_markdown(rows: list[dict[str, object]], ablations: list[dict[str, object]], runtime_seconds: float) -> str:
    means = _method_means(rows)
    lines = [
        "# Stage 3B Incremental Structural Enhancement Analysis",
        "",
        f"Runtime seconds: {runtime_seconds:.2f}",
        "",
        "## Overall Mean NMSE",
    ]
    for method, value in sorted(means.items(), key=lambda item: item[1]):
        lines.append(f"- {method}: {value:.8f}")
    lookup = {row["question"]: row for row in ablations}
    lines.extend(
        [
            "",
            "## Required Questions",
            f"1. Structural enhancement improves PICASSO-rec: {_yes_no_delta(lookup, 'architecture')}.",
            f"2. Physics loss remains effective: {_yes_no_delta(lookup, 'physics')}.",
            f"3. Feature-level GAN beats output/supervised physics: {_yes_no_delta(lookup, 'feature_gan')}.",
            f"4. Attention helps: {_yes_no_delta(lookup, 'attention')}.",
            f"5. Condition-aware FiLM helps: {_yes_no_delta(lookup, 'condition')}.",
            "6. Bottleneck judgment: compare the ablations above; if architecture improves more than physics/GAN, bottleneck is architecture. If none improve materially, data complexity is likely limiting.",
            "",
            "## Final Bottleneck Judgment",
            _bottleneck_judgment(lookup),
        ]
    )
    return "\n".join(lines) + "\n"


def _yes_no_delta(lookup: dict[str, dict[str, object]], key: str) -> str:
    if key not in lookup:
        return "not available"
    delta = float(lookup[key]["delta_nmse"])
    return f"{'yes' if delta < 0 else 'no'} (delta={delta:.8f})"


def _bottleneck_judgment(lookup: dict[str, dict[str, object]]) -> str:
    architecture = float(lookup.get("architecture", {}).get("delta_nmse", 0.0))
    physics = float(lookup.get("physics", {}).get("delta_nmse", 0.0))
    gan = float(lookup.get("feature_gan", {}).get("delta_nmse", 0.0))
    if architecture < physics and architecture < gan:
        return "The strongest evidence points to architecture/expression capacity as the current bottleneck."
    if physics < architecture and physics < gan:
        return "The strongest evidence points to physics constraints and weighting as the current bottleneck."
    if gan < architecture and gan < physics:
        return "The strongest evidence points to GAN design as the current bottleneck."
    return "No module gives a decisive gain; data complexity is likely becoming the bottleneck."


def _method_means(rows: list[dict[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["nmse"]))
    return {method: mean(values) for method, values in grouped.items()}


def _make_row(config: dict[str, Any], spec: dict[str, Any], seed: int, pilot_ratio: float, snr_db: float, metrics: dict[str, float], runtime_seconds: float) -> dict[str, object]:
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
        "epochs_used": 0 if spec["kind"] == "ls" else config["training"]["epochs"],
        "loss_mode": spec["loss_mode"],
        "use_attention": spec["attention"],
        "use_condition": spec["condition"],
        "feature_discriminator": spec["feature_disc"],
        "runtime_seconds": f"{runtime_seconds:.4f}",
    }


def _clip(model: torch.nn.Module, config: dict[str, Any]) -> None:
    max_norm = float(config["training"].get("gradient_clip_norm", 0.0))
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _validate_config(config: dict[str, Any]) -> None:
    if bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 3B must not save generated datasets.")
    policy = config["artifact_policy"]
    if bool(policy.get("save_checkpoints", False)) or bool(policy.get("save_numpy_arrays", False)):
        raise ValueError("Stage 3B incremental task must remain lightweight.")


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
        raise ValueError(f"Config {path!r} must contain a YAML mapping.")
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3B incremental PICASSO enhancements.")
    parser.add_argument("--config", default="configs/stage3b_incremental.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
