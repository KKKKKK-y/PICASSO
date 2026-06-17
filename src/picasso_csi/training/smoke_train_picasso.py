"""Stage 1C smoke training loop for PICASSO skeleton models."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import SyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import nmse, pilot_consistency_error  # noqa: E402
from picasso_csi.losses import discriminator_bce_loss, picasso_generator_loss  # noqa: E402
from picasso_csi.models import PICASSODiscriminator, PICASSOGenerator  # noqa: E402


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_smoke_training(config)
    print(f"device: {result['device']}")
    print(f"train samples: {result['train_samples']}")
    print(f"val samples: {result['val_samples']}")
    print(f"final generator loss: {result['generator_loss']:.6f}")
    print(f"final discriminator loss: {result['discriminator_loss']:.6f}")
    print(f"final validation NMSE: {result['validation_nmse']:.6f}")
    print(f"final pilot consistency error: {result['pilot_consistency_error']:.6f}")
    print("PICASSO smoke training completed")


def run_smoke_training(config: dict[str, Any]) -> dict[str, Any]:
    _validate_smoke_config(config)
    data_config = config["data"]
    system_config = config["system"]
    training_config = config["training"]
    model_config = config["model"]
    seed = int(data_config.get("seed", 42))
    _set_seed(seed)

    device = _resolve_device(str(training_config.get("device", "cuda")))
    num_samples = int(data_config["num_samples"])
    val_samples = max(1, min(64, num_samples // 4))
    train_dataset = _build_dataset(data_config, system_config, num_samples, seed)
    val_dataset = _build_dataset(data_config, system_config, val_samples, seed + num_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    generator = PICASSOGenerator(**model_config["generator"]).to(device)
    discriminator = PICASSODiscriminator(**model_config["discriminator"]).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training_config["learning_rate_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training_config["learning_rate_d"]))

    latest = {
        "generator_loss": float("nan"),
        "discriminator_loss": float("nan"),
        "validation_nmse": float("nan"),
        "pilot_consistency_error": float("nan"),
    }
    for epoch in range(1, int(training_config["smoke_epochs"]) + 1):
        generator_loss, discriminator_loss = _train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            device,
            config["loss"],
        )
        validation_nmse, pilot_error = _evaluate_generator(generator, val_loader, device)
        latest = {
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "validation_nmse": validation_nmse,
            "pilot_consistency_error": pilot_error,
        }
        print(
            f"epoch {epoch}/{training_config['smoke_epochs']} - "
            f"generator loss: {generator_loss:.6f} - "
            f"discriminator loss: {discriminator_loss:.6f} - "
            f"val NMSE: {validation_nmse:.6f} - "
            f"pilot consistency: {pilot_error:.6f}"
        )

    return {
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        **latest,
    }


def _train_one_epoch(
    generator: PICASSOGenerator,
    discriminator: PICASSODiscriminator,
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    loss_config: dict[str, Any],
) -> tuple[float, float]:
    generator.train()
    discriminator.train()
    total_g = 0.0
    total_d = 0.0
    total_samples = 0
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        batch_size = H_full.shape[0]

        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            H_fake_detached = generator(H_sparse)
        real_logits = discriminator(H_full)
        fake_logits = discriminator(H_fake_detached.detach())
        d_loss = discriminator_bce_loss(real_logits, fake_logits)
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad(set_to_none=True)
        H_hat = generator(H_sparse)
        fake_logits_for_g = discriminator(H_hat)
        losses = picasso_generator_loss(
            H_hat,
            H_full,
            H_sparse,
            mask,
            fake_logits=fake_logits_for_g,
            lambda_rec=float(loss_config["lambda_rec"]),
            lambda_adv=float(loss_config["lambda_adv"]),
            lambda_pilot=float(loss_config["lambda_pilot"]),
            lambda_smooth=float(loss_config["lambda_smooth"]),
            lambda_sparse=float(loss_config["lambda_sparse"]),
        )
        g_loss = losses["total"]
        g_loss.backward()
        optimizer_g.step()

        total_g += float(g_loss.detach().cpu()) * batch_size
        total_d += float(d_loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_g / max(total_samples, 1), total_d / max(total_samples, 1)


@torch.no_grad()
def _evaluate_generator(
    generator: PICASSOGenerator,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    generator.eval()
    total_nmse = 0.0
    total_pilot = 0.0
    total_samples = 0
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        H_hat = generator(H_sparse)
        batch_size = H_full.shape[0]
        total_nmse += float(nmse(H_hat, H_full).detach().cpu()) * batch_size
        total_pilot += float(pilot_consistency_error(H_hat, H_sparse, mask).detach().cpu()) * batch_size
        total_samples += batch_size
    return total_nmse / max(total_samples, 1), total_pilot / max(total_samples, 1)


def _build_dataset(
    data_config: dict[str, Any],
    system_config: dict[str, Any],
    num_samples: int,
    seed: int,
) -> SyntheticCSIDataset:
    return SyntheticCSIDataset(
        num_samples=num_samples,
        n_tx=int(system_config["n_tx"]),
        n_rx=int(system_config["n_rx"]),
        n_subcarriers=int(system_config["n_subcarriers"]),
        n_paths=int(system_config["n_paths"]),
        pilot_ratio=float(system_config["pilot_ratio"]),
        seed=seed,
    )


def _validate_smoke_config(config: dict[str, Any]) -> None:
    data_config = config["data"]
    training_config = config["training"]
    policy = config.get("artifact_policy", {})
    if bool(data_config.get("save_generated_data", False)):
        raise ValueError("Stage 1C smoke training must not save generated data.")
    if int(data_config["num_samples"]) > 256:
        raise ValueError("Stage 1C smoke training caps num_samples at 256.")
    epochs = int(training_config["smoke_epochs"])
    if epochs <= 0 or epochs > 2:
        raise ValueError("Stage 1C smoke training requires 1 or 2 epochs.")
    blocked = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Stage 1C smoke training must not save artifacts: {enabled}.")


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
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 1C smoke training.")
    parser.add_argument("--config", default="configs/stage1c_picasso_smoke.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
