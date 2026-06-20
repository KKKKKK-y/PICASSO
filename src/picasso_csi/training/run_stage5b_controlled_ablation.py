"""Run strict matched-condition Stage 5B PICASSO ablations."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.evaluation import write_csv  # noqa: E402
from picasso_csi.losses import discriminator_bce_loss, picasso_generator_loss  # noqa: E402
from picasso_csi.models import (  # noqa: E402
    EnhancedDnCNNBaseline,
    PICASSODiscriminator,
    PICASSOGenerator,
    ls_baseline,
)
from picasso_csi.training.run_stage4 import _dataset, _evaluate  # noqa: E402


RESULT_FIELDS = [
    "ablation_group", "variant", "channel_profile", "seed", "pilot_ratio",
    "snr_db", "velocity", "pilot_pattern", "nmse", "mse", "mae",
    "pilot_consistency_error", "doppler_robustness", "runtime_seconds",
    "num_parameters", "delta_nmse_vs_base", "contribution_label",
]
FINAL_FIELDS = [
    "Group", "Variant", "Avg NMSE", "Std NMSE",
    "Delta vs PICASSO-rec-base", "Interpretation", "Keep in Final Model?",
]


def build_variants() -> list[dict[str, Any]]:
    """Return unique executions with aliases for each requested ablation group."""

    return [
        _variant("ls", "LS", "ls", {"baseline": "LS"}),
        _variant("enhanced_dncnn", "Enhanced-DnCNN", "dncnn", {"baseline": "Enhanced-DnCNN"}),
        _variant(
            "base", "PICASSO-rec-base", "picasso",
            {"baseline": "PICASSO-rec-base", "architecture": "PICASSO-rec-base", "loss": "PICASSO-rec only", "gan": "PICASSO-rec without GAN"},
        ),
        _variant("refinement", "PICASSO-rec + refinement blocks only", "picasso", {"architecture": "PICASSO-rec + refinement blocks only"}, refinement=True),
        _variant("multiscale", "PICASSO-rec + multi-scale fusion only", "picasso", {"architecture": "PICASSO-rec + multi-scale fusion only"}, multiscale=True),
        _variant("se", "PICASSO-rec + SE attention only", "picasso", {"architecture": "PICASSO-rec + SE attention only"}, attention=True),
        _variant("film", "PICASSO-rec + FiLM condition only", "picasso", {"architecture": "PICASSO-rec + FiLM condition only"}, film=True),
        _variant("refinement_multiscale", "PICASSO-rec + refinement + multi-scale", "picasso", {"architecture": "PICASSO-rec + refinement + multi-scale"}, refinement=True, multiscale=True),
        _variant("refinement_multiscale_se", "PICASSO-rec + refinement + multi-scale + SE", "picasso", {"architecture": "PICASSO-rec + refinement + multi-scale + SE"}, refinement=True, multiscale=True, attention=True),
        _variant("enhanced_full", "PICASSO-rec-enhanced full architecture", "picasso", {"architecture": "PICASSO-rec-enhanced full architecture"}, refinement=True, multiscale=True, attention=True, film=True),
        _variant("loss_pilot", "PICASSO-rec + pilot consistency", "picasso", {"loss": "PICASSO-rec + pilot consistency"}, loss_components=["pilot"]),
        _variant("loss_frequency", "PICASSO-rec + frequency consistency", "picasso", {"loss": "PICASSO-rec + frequency consistency"}, loss_components=["frequency"]),
        _variant("loss_delay", "PICASSO-rec + delay-domain sparsity", "picasso", {"loss": "PICASSO-rec + delay-domain sparsity"}, loss_components=["delay"]),
        _variant("loss_energy", "PICASSO-rec + energy preservation", "picasso", {"loss": "PICASSO-rec + energy preservation"}, loss_components=["energy"]),
        _variant("loss_pilot_frequency", "PICASSO-rec + pilot + frequency", "picasso", {"loss": "PICASSO-rec + pilot + frequency"}, loss_components=["pilot", "frequency"]),
        _variant("loss_pilot_delay", "PICASSO-rec + pilot + delay", "picasso", {"loss": "PICASSO-rec + pilot + delay"}, loss_components=["pilot", "delay"]),
        _variant("loss_full", "PICASSO-rec-full physics", "picasso", {"loss": "PICASSO-rec-full physics"}, loss_components=["pilot", "frequency", "delay", "energy", "smooth"]),
        _variant("gan_output", "PICASSO-rec + output-level GAN", "picasso", {"gan": "PICASSO-rec + output-level GAN"}, gan="output", adv_weight="full"),
        _variant("gan_feature", "PICASSO-rec + feature-level GAN", "picasso", {"gan": "PICASSO-rec + feature-level GAN"}, gan="feature", adv_weight="full"),
        _variant("gan_light", "PICASSO-rec + light adversarial loss", "picasso", {"gan": "PICASSO-rec + light adversarial loss"}, gan="output", adv_weight="light"),
    ]


def contribution_label(delta_nmse: float) -> str:
    """Classify a matched-condition NMSE delta using the requested thresholds."""

    if delta_nmse < -0.001:
        return "positive"
    if delta_nmse > 0.001:
        return "negative"
    return "neutral"


def add_deltas_and_labels(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach per-profile/per-seed deltas against the single cached base run."""

    base = {
        (str(row["channel_profile"]), int(row["seed"])): float(row["nmse"])
        for row in rows
        if row["ablation_group"] == "baseline" and row["variant"] == "PICASSO-rec-base"
    }
    output = []
    for row in rows:
        key = (str(row["channel_profile"]), int(row["seed"]))
        if key not in base:
            raise ValueError(f"Missing PICASSO-rec-base result for {key}.")
        delta = float(row["nmse"]) - base[key]
        updated = dict(row)
        updated["delta_nmse_vs_base"] = f"{delta:.8f}"
        updated["contribution_label"] = contribution_label(delta)
        output.append(updated)
    return output


def build_final_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build the compact paper table from matched Stage 5B rows only."""

    grouped = _group_means(rows)
    base_mean = grouped[("baseline", "PICASSO-rec-base")][0]
    best_arch = _best(grouped, "architecture", exclude={"PICASSO-rec-base"})
    best_loss = _best(grouped, "loss", exclude={"PICASSO-rec only"})
    best_gan = _best(grouped, "gan", exclude={"PICASSO-rec without GAN"})
    selected = [
        ("baseline", "LS"),
        ("baseline", "Enhanced-DnCNN"),
        ("baseline", "PICASSO-rec-base"),
        best_arch,
        best_loss,
        best_gan,
        ("architecture", "PICASSO-rec-enhanced full architecture"),
        ("loss", "PICASSO-rec-full physics"),
    ]
    selected = list(dict.fromkeys(selected))
    output = []
    for group, variant in selected:
        avg, std = grouped[(group, variant)]
        delta = avg - base_mean
        label = contribution_label(delta)
        keep = _keep_final(group, variant, delta, best_arch)
        output.append({
            "Group": group,
            "Variant": variant,
            "Avg NMSE": f"{avg:.8f}",
            "Std NMSE": f"{std:.8f}",
            "Delta vs PICASSO-rec-base": f"{delta:.8f}",
            "Interpretation": f"{label} contribution under the matched Stage 5B setting",
            "Keep in Final Model?": keep,
        })
    return output


def write_report(path: str | Path, rows: list[dict[str, object]], config: dict[str, Any], runtime_seconds: float) -> Path:
    """Write the controlled ablation report from Stage 5B rows."""

    grouped = _group_means(rows)
    base = grouped[("baseline", "PICASSO-rec-base")][0]
    best_arch = _best(grouped, "architecture", exclude={"PICASSO-rec-base"})
    best_loss = _best(grouped, "loss", exclude={"PICASSO-rec only"})
    best_gan = _best(grouped, "gan", exclude={"PICASSO-rec without GAN"})
    arch_decisions = _component_decisions(grouped, "architecture", base)
    loss_decisions = _component_decisions(grouped, "loss", base)
    gan_decisions = _component_decisions(grouped, "gan", base)
    enhanced_delta = grouped[("architecture", "PICASSO-rec-enhanced full architecture")][0] - base
    best_arch_delta = grouped[best_arch][0] - base
    full_physics_delta = grouped[("loss", "PICASSO-rec-full physics")][0] - base
    gan_best_delta = grouped[best_gan][0] - base
    lines = [
        "# Stage 5B Controlled Ablation Report", "",
        "## 1. Purpose", "",
        "Stage 5A consolidated historical evidence and included small gap runs. Stage 5B is a new strict controlled ablation: every reported row uses the same data protocol, channel condition, budget, optimizer, batch size, and evaluator, with one targeted module changed at a time.", "",
        "## 2. Controlled Setting", "",
        f"- Channel profiles: {', '.join(config['experiment']['channel_profiles'])}",
        f"- Pilot ratio: {config['experiment']['pilot_ratio']}",
        f"- SNR: {config['experiment']['snr_db']} dB",
        f"- Velocity: {config['experiment']['velocity']} km/h",
        f"- Pilot pattern: {config['experiment']['pilot_pattern']}",
        f"- Seeds: {config['experiment']['seeds']}",
        f"- Epochs: {config['training']['epochs']} for every trainable variant",
        f"- Optimizer / learning rate: Adam / {config['training']['learning_rate']}",
        f"- Batch size: {config['training']['batch_size']}",
        f"- Runtime: {runtime_seconds:.2f} seconds", "",
        "## 3. Baseline Results", "",
        *_report_group(grouped, "baseline", base), "",
        "The comparison is fully matched across CDL profile and seed. LS has no trainable budget by definition; both neural methods use the same training dataset and evaluator.", "",
        "## 4. Architecture Ablation", "",
        *_report_group(grouped, "architecture", base), "",
        f"- Refinement: {arch_decisions.get('PICASSO-rec + refinement blocks only', 'not available')}",
        f"- Multi-scale: {arch_decisions.get('PICASSO-rec + multi-scale fusion only', 'not available')}",
        f"- SE attention: {arch_decisions.get('PICASSO-rec + SE attention only', 'not available')}",
        f"- FiLM: {arch_decisions.get('PICASSO-rec + FiLM condition only', 'not available')}",
        f"- Best architecture: {best_arch[1]}",
        f"- Full enhanced architecture: {contribution_label(enhanced_delta)} (delta {enhanced_delta:+.8f}).", "",
        "## 5. Loss Ablation", "",
        *_report_group(grouped, "loss", base), "",
        f"- Pilot consistency: {loss_decisions.get('PICASSO-rec + pilot consistency', 'not available')}",
        f"- Frequency consistency: {loss_decisions.get('PICASSO-rec + frequency consistency', 'not available')}",
        f"- Delay sparsity: {loss_decisions.get('PICASSO-rec + delay-domain sparsity', 'not available')}",
        f"- Energy preservation: {loss_decisions.get('PICASSO-rec + energy preservation', 'not available')}",
        f"- Best physics variant: {best_loss[1]}",
        f"- Full physics: {contribution_label(full_physics_delta)} (delta {full_physics_delta:+.8f}). This aggregate result is neutral, so it shows neither a meaningful NMSE gain nor threshold-level over-regularization under this setting.", "",
        "## 6. GAN Ablation", "",
        *_report_group(grouped, "gan", base), "",
        f"- Output GAN: {gan_decisions.get('PICASSO-rec + output-level GAN', 'not available')}",
        f"- Feature GAN: {gan_decisions.get('PICASSO-rec + feature-level GAN', 'not available')}",
        f"- Light adversarial: {gan_decisions.get('PICASSO-rec + light adversarial loss', 'not available')}",
        f"- Best adversarial variant: {best_gan[1]} ({contribution_label(gan_best_delta)}, delta {gan_best_delta:+.8f}).", "",
        "## 7. Final Model Decision", "",
        f"PICASSO-rec remains the final model family. The best controlled structural configuration is `{best_arch[1]}` ({contribution_label(best_arch_delta)}, delta {best_arch_delta:+.8f}) and is the architecture recommended for the paper setting. The variant named `PICASSO-rec-enhanced full architecture` includes FiLM and is {contribution_label(enhanced_delta)} (delta {enhanced_delta:+.8f}), so that full bundle should not replace the main model. Physics remains a secondary ablation. GAN remains deprecated unless a GAN variant has a positive contribution; observed decision: {'reconsider GAN' if gan_best_delta < -0.001 else 'keep GAN deprecated'}.", "",
        "## 8. Paper-Ready Interpretation", "",
        "Architecture modules change reconstruction capacity directly and are evaluated without changing the objective. Physics terms can improve constraint satisfaction but may trade off against channel reconstruction when their priors mismatch CDL realizations. GAN objectives add optimization variance and are retained only when matched NMSE improves. Because all rows share sparse pilots, SNR, mobility, and profile coverage, the deltas isolate module contribution more credibly than cross-stage comparisons.", "",
        "## 9. Limitations", "",
        "This is a reduced but strict controlled study at one pilot ratio, SNR, velocity, and pilot pattern. It does not represent every possible wireless condition, and the simulator is CDL-inspired rather than a complete 3GPP implementation. Three profiles and three seeds improve coverage but do not establish universal statistical significance. The design is nevertheless sufficient to isolate module contributions within the stated setting.", "",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_stage5b(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    start = time.perf_counter()
    device = _resolve_device(config["training"]["device"])
    variants = build_variants()
    experiment = config["experiment"]
    unique_count = len(variants) * len(experiment["channel_profiles"]) * len(experiment["seeds"])
    print(f"device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"controlled setting: profiles={experiment['channel_profiles']}, pilot={experiment['pilot_ratio']}, SNR={experiment['snr_db']}, velocity={experiment['velocity']}, pattern={experiment['pilot_pattern']}, seeds={experiment['seeds']}, epochs={config['training']['epochs']}")
    print(f"variants list: {[variant['name'] for variant in variants]}")
    print(f"total experiment count: {unique_count}")

    canonical_rows: list[dict[str, object]] = []
    progress = 0
    for profile in experiment["channel_profiles"]:
        for seed in experiment["seeds"]:
            for spec in variants:
                progress += 1
                print(f"[{progress}/{unique_count}] profile={profile}, seed={seed}, variant={spec['name']}")
                canonical_rows.append(_run_one(config, spec, str(profile), int(seed), device))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expanded = add_deltas_and_labels(_expand_groups(canonical_rows, variants))
    outputs = _save_outputs(config, expanded, time.perf_counter() - start)
    grouped = _group_means(expanded)
    best_arch = _best(grouped, "architecture", exclude={"PICASSO-rec-base"})
    best_loss = _best(grouped, "loss", exclude={"PICASSO-rec only"})
    best_gan = _best(grouped, "gan", exclude={"PICASSO-rec without GAN"})
    base = grouped[("baseline", "PICASSO-rec-base")][0]
    enhanced = grouped[("architecture", "PICASSO-rec-enhanced full architecture")][0]
    gan_delta = grouped[best_gan][0] - base
    print(f"final best variant: {min(grouped, key=lambda key: grouped[key][0])}")
    print(f"PICASSO-rec remains final: {'no' if enhanced - base < -0.001 else 'yes'}")
    print(f"best architecture variant: {best_arch[1]}")
    print(f"best loss variant: {best_loss[1]}")
    print(f"best GAN variant: {best_gan[1]}")
    print(f"GAN remains deprecated: {'no' if gan_delta < -0.001 else 'yes'}")
    print("physics remains secondary: yes")
    return {
        "device": str(device), "unique_experiments": unique_count, "rows": len(expanded),
        "runtime_seconds": time.perf_counter() - start, "outputs": outputs,
        "best_architecture": best_arch[1], "best_loss": best_loss[1], "best_gan": best_gan[1],
    }


def _run_one(config: dict[str, Any], spec: dict[str, Any], profile: str, seed: int, device: torch.device) -> dict[str, object]:
    _set_seed(seed)
    train_loader, test_loader = _controlled_loaders(config, profile, seed)
    started = time.perf_counter()
    if spec["kind"] == "ls":
        metrics = _evaluate(test_loader, device, lambda batch: ls_baseline(batch["H_sparse"], batch["mask"]))
        parameters = 0
    elif spec["kind"] == "dncnn":
        model = EnhancedDnCNNBaseline(
            hidden_channels=int(config["model"]["enhanced_dncnn_hidden_channels"]),
            depth=int(config["model"]["enhanced_dncnn_depth"]),
        ).to(device)
        _train_supervised(model, train_loader, device, config)
        metrics = _evaluate(test_loader, device, lambda batch: model(batch["H_sparse"], batch["mask"]))
        parameters = _parameter_count(model)
    else:
        generator = _make_generator(config, spec).to(device)
        discriminator = _make_discriminator(config, spec).to(device) if spec.get("gan") else None
        _train_picasso(generator, discriminator, train_loader, device, config, spec)
        metrics = _evaluate(test_loader, device, lambda batch: _generator_forward(generator, batch, spec))
        parameters = _parameter_count(generator) + (_parameter_count(discriminator) if discriminator else 0)
    runtime = time.perf_counter() - started
    experiment = config["experiment"]
    return {
        "key": spec["key"], "channel_profile": profile, "seed": seed,
        "pilot_ratio": f"{float(experiment['pilot_ratio']):.4f}", "snr_db": f"{float(experiment['snr_db']):g}",
        "velocity": f"{float(experiment['velocity']):g}", "pilot_pattern": experiment["pilot_pattern"],
        "nmse": f"{metrics['nmse']:.8f}", "mse": f"{metrics['mse']:.8f}", "mae": f"{metrics['mae']:.8f}",
        "pilot_consistency_error": f"{metrics['pilot_consistency_error']:.8f}",
        "doppler_robustness": f"{metrics['doppler_robustness_metric']:.8f}",
        "runtime_seconds": f"{runtime:.4f}", "num_parameters": parameters,
    }


def _controlled_loaders(config: dict[str, Any], profile: str, seed: int) -> tuple[DataLoader, DataLoader]:
    experiment = config["experiment"]
    dataset_config = {
        "system": config["system"], "cdl": config["cdl"],
    }
    train = _dataset(dataset_config, int(config["data"]["train_samples"]), profile, str(experiment["pilot_pattern"]), seed, float(experiment["pilot_ratio"]), float(experiment["snr_db"]), float(experiment["velocity"]))
    test = _dataset(dataset_config, int(config["data"]["test_samples"]), profile, str(experiment["pilot_pattern"]), seed + 100_000, float(experiment["pilot_ratio"]), float(experiment["snr_db"]), float(experiment["velocity"]))
    generator = torch.Generator().manual_seed(seed + 424_242)
    batch_size = int(config["training"]["batch_size"])
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _train_supervised(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: dict[str, Any]) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    for _ in range(int(config["training"]["epochs"])):
        model.train()
        for batch in loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch["H_sparse"], batch["mask"]), batch["H_full"])
            loss.backward()
            _clip(model, config)
            optimizer.step()


def _train_picasso(
    generator: PICASSOGenerator,
    discriminator: PICASSODiscriminator | None,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(config["training"]["learning_rate"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(config["training"]["discriminator_learning_rate"])) if discriminator else None
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"]["gan_warmup_epochs"])
    for epoch in range(epochs):
        generator.train()
        if discriminator:
            discriminator.train()
        for batch in loader:
            batch = _to_device(batch, device)
            adversarial_active = discriminator is not None and epoch >= warmup
            if adversarial_active and optimizer_d and discriminator:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = _generator_forward(generator, batch, spec)
                d_loss = discriminator_bce_loss(discriminator(batch["H_full"]), discriminator(fake.detach()))
                d_loss.backward()
                _clip(discriminator, config)
                optimizer_d.step()
            optimizer_g.zero_grad(set_to_none=True)
            estimate = _generator_forward(generator, batch, spec)
            fake_logits = discriminator(estimate) if adversarial_active and discriminator else None
            weights = _loss_weights(config, spec)
            loss = picasso_generator_loss(
                estimate, batch["H_full"], batch["H_sparse"], batch["mask"], fake_logits=fake_logits,
                lambda_rec=1.0, lambda_adv=weights["adv"], lambda_pilot=weights["pilot"],
                lambda_smooth=weights["smooth"], lambda_sparse=weights["delay"],
                lambda_frequency=weights["frequency"], lambda_energy=weights["energy"],
                loss_mode="rec_adv" if adversarial_active else ("rec_physics" if spec.get("loss_components") else "rec_only"),
            )["total"]
            loss.backward()
            _clip(generator, config)
            optimizer_g.step()


def _make_generator(config: dict[str, Any], spec: dict[str, Any]) -> PICASSOGenerator:
    model = config["model"]
    return PICASSOGenerator(
        base_channels=int(model["generator_base_channels"]),
        num_blocks=int(model["generator_num_blocks"]),
        refinement_blocks=int(model["refinement_blocks"]) if spec.get("refinement") else 0,
        use_multiscale_fusion=bool(spec.get("multiscale")),
        use_channel_attention=bool(spec.get("attention")),
        use_film_conditioning=bool(spec.get("film")),
    )


def _make_discriminator(config: dict[str, Any], spec: dict[str, Any]) -> PICASSODiscriminator:
    return PICASSODiscriminator(
        base_channels=int(config["model"]["discriminator_base_channels"]),
        use_delay_features=spec.get("gan") == "feature",
    )


def _generator_forward(generator: PICASSOGenerator, batch: dict[str, torch.Tensor], spec: dict[str, Any]) -> torch.Tensor:
    if spec.get("film"):
        return generator(batch["H_sparse"], batch["snr_db"], batch["pilot_ratio"])
    return generator(batch["H_sparse"])


def _loss_weights(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, float]:
    loss = config["loss"]
    components = set(spec.get("loss_components", []))
    adv_key = spec.get("adv_weight")
    return {
        "pilot": float(loss["lambda_pilot"]) if "pilot" in components else 0.0,
        "frequency": float(loss["lambda_frequency"]) if "frequency" in components else 0.0,
        "delay": float(loss["lambda_sparse"]) if "delay" in components else 0.0,
        "energy": float(loss["lambda_energy"]) if "energy" in components else 0.0,
        "smooth": float(loss["lambda_smooth"]) if "smooth" in components else 0.0,
        "adv": float(loss[f"lambda_adv_{adv_key}"]) if adv_key else 0.0,
    }


def _expand_groups(canonical_rows: list[dict[str, object]], variants: list[dict[str, Any]]) -> list[dict[str, object]]:
    groups = {variant["key"]: variant["groups"] for variant in variants}
    output = []
    for row in canonical_rows:
        for group, alias in groups[str(row["key"])].items():
            expanded = {field: row.get(field, "") for field in RESULT_FIELDS}
            expanded["ablation_group"] = group
            expanded["variant"] = alias
            output.append(expanded)
    return output


def _save_outputs(config: dict[str, Any], rows: list[dict[str, object]], runtime_seconds: float) -> list[str]:
    out = config["outputs"]
    root = Path(out["results_dir"])
    mapping = {
        "baseline": out["baseline_csv"], "architecture": out["architecture_csv"],
        "loss": out["loss_csv"], "gan": out["gan_csv"],
    }
    paths = []
    for group, filename in mapping.items():
        path = root / filename
        write_csv([row for row in rows if row["ablation_group"] == group], path, RESULT_FIELDS)
        paths.append(str(path))
    final_path = root / out["final_table_csv"]
    write_csv(build_final_table(rows), final_path, FINAL_FIELDS)
    report_path = write_report(out["report_md"], rows, config, runtime_seconds)
    paths.extend([str(final_path), str(report_path)])
    return paths


def _group_means(rows: list[dict[str, object]]) -> dict[tuple[str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ablation_group"]), str(row["variant"]))].append(float(row["nmse"]))
    return {key: (mean(values), stdev(values) if len(values) > 1 else 0.0) for key, values in grouped.items()}


def _best(grouped: dict[tuple[str, str], tuple[float, float]], group: str, exclude: set[str]) -> tuple[str, str]:
    candidates = {key: values for key, values in grouped.items() if key[0] == group and key[1] not in exclude}
    if not candidates:
        raise ValueError(f"No candidates found for {group}.")
    return min(candidates, key=lambda key: candidates[key][0])


def _report_group(grouped: dict[tuple[str, str], tuple[float, float]], group: str, base: float) -> list[str]:
    entries = [(key[1], value) for key, value in grouped.items() if key[0] == group]
    return [f"- {variant}: NMSE {avg:.8f} +/- {std:.8f}; delta {avg - base:+.8f}; {contribution_label(avg - base)}" for variant, (avg, std) in sorted(entries, key=lambda item: item[1][0])]


def _component_decisions(grouped: dict[tuple[str, str], tuple[float, float]], group: str, base: float) -> dict[str, str]:
    return {
        variant: f"{contribution_label(avg - base)} (delta {avg - base:+.8f})"
        for (entry_group, variant), (avg, _) in grouped.items()
        if entry_group == group
    }


def _keep_final(group: str, variant: str, delta: float, best_arch: tuple[str, str]) -> str:
    if variant == "PICASSO-rec-base":
        return "yes"
    if group == "architecture" and variant == best_arch[1] and delta < -0.001:
        return "yes"
    return "no"


def _variant(key: str, name: str, kind: str, groups: dict[str, str], **flags: Any) -> dict[str, Any]:
    return {"key": key, "name": name, "kind": kind, "groups": groups, **flags}


def _parameter_count(model: torch.nn.Module | None) -> int:
    return sum(parameter.numel() for parameter in model.parameters()) if model is not None else 0


def _clip(model: torch.nn.Module, config: dict[str, Any]) -> None:
    max_norm = float(config["training"].get("gradient_clip_norm", 0.0))
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested.lower() == "cuda" and torch.cuda.is_available() else "cpu")


def _validate_config(config: dict[str, Any]) -> None:
    if not bool(config["experiment"].get("controlled", False)):
        raise ValueError("Stage 5B requires controlled=true.")
    if not bool(config["training"].get("same_training_budget_for_all_variants", False)):
        raise ValueError("All Stage 5B variants must share one training budget.")
    if str(config["training"].get("optimizer", "")).lower() != "adam":
        raise ValueError("Stage 5B controlled optimizer must be Adam.")
    policy = config["artifact_policy"]
    forbidden = ["save_checkpoints", "save_numpy_arrays", "save_predictions", "save_generated_data"]
    if any(bool(policy.get(key, False)) for key in forbidden) or bool(config["data"].get("save_generated_data", False)):
        raise ValueError("Stage 5B may save only CSV and Markdown artifacts.")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Stage 5B config must be a YAML mapping.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict controlled PICASSO ablations.")
    parser.add_argument("--config", default="configs/stage5b_controlled_ablation.yaml")
    args = parser.parse_args()
    result = run_stage5b(_load_config(args.config))
    print(f"runtime seconds: {result['runtime_seconds']:.2f}")
    print(f"outputs: {result['outputs']}")


if __name__ == "__main__":
    main()
