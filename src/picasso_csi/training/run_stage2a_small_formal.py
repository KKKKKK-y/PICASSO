"""Run Stage 2A noise-aware small formal experiment."""

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

from picasso_csi.datasets import NoisySyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import (  # noqa: E402
    delay_domain_sparsity_score,
    mae,
    mse,
    nmse,
    pilot_consistency_error,
    write_result_csv,
)
from picasso_csi.losses import discriminator_bce_loss, picasso_generator_loss  # noqa: E402
from picasso_csi.models import DnCNNBaseline, PICASSODiscriminator, PICASSOGenerator, ls_baseline  # noqa: E402


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_stage2a(config)
    print(f"device: {result['device']}")
    print(f"csv path: {result['csv_path']}")
    print(f"rows: {result['rows']}")
    print("Stage 2A small formal experiment completed")


def run_stage2a(config: dict[str, Any]) -> dict[str, Any]:
    _validate_config(config)
    data_config = config["data"]
    system_config = config["system"]
    training_config = config["training"]
    outputs_config = config["outputs"]
    seed = int(data_config["seed"])
    _set_seed(seed)
    device = _resolve_device(str(training_config.get("device", "cuda")))
    rows: list[dict[str, object]] = []

    for pilot_ratio in data_config["pilot_ratios"]:
        for snr_db in data_config["snr_db_values"]:
            pilot_ratio_f = float(pilot_ratio)
            snr_db_f = float(snr_db)
            print(f"running pilot_ratio={pilot_ratio_f:.4f}, snr_db={snr_db_f:g}")
            train_loader, test_loader = _build_loaders(
                data_config,
                system_config,
                pilot_ratio=pilot_ratio_f,
                snr_db=snr_db_f,
                seed=seed,
            )

            if bool(config["models"].get("run_ls", True)):
                metrics = _evaluate_estimator(
                    test_loader,
                    device,
                    lambda H_sparse, mask: ls_baseline(H_sparse, mask),
                )
                rows.append(_make_row(config, "LS", pilot_ratio_f, snr_db_f, metrics, epochs=0))
                print(f"  LS NMSE: {metrics['nmse']:.6f}")

            if bool(config["models"].get("run_dncnn", True)):
                dncnn = DnCNNBaseline().to(device)
                _train_dncnn(
                    dncnn,
                    train_loader,
                    device,
                    epochs=int(training_config["epochs"]),
                    learning_rate=float(training_config["learning_rate_g"]),
                )
                metrics = _evaluate_estimator(test_loader, device, lambda H_sparse, mask: dncnn(H_sparse, mask))
                rows.append(_make_row(config, "DnCNN", pilot_ratio_f, snr_db_f, metrics, int(training_config["epochs"])))
                print(f"  DnCNN NMSE: {metrics['nmse']:.6f}")

            if bool(config["models"].get("run_picasso", True)):
                generator = PICASSOGenerator().to(device)
                discriminator = PICASSODiscriminator().to(device)
                _train_picasso(
                    generator,
                    discriminator,
                    train_loader,
                    device,
                    config,
                )
                metrics = _evaluate_estimator(test_loader, device, lambda H_sparse, mask: generator(H_sparse))
                rows.append(
                    _make_row(config, "PICASSO", pilot_ratio_f, snr_db_f, metrics, int(training_config["epochs"]))
                )
                print(f"  PICASSO NMSE: {metrics['nmse']:.6f}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

    output_path = Path(outputs_config["results_dir"]) / str(outputs_config["result_csv"])
    write_result_csv(rows, output_path)
    return {
        "device": str(device),
        "csv_path": str(output_path),
        "rows": len(rows),
    }


def _build_loaders(
    data_config: dict[str, Any],
    system_config: dict[str, Any],
    pilot_ratio: float,
    snr_db: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = _build_dataset(
        data_config,
        system_config,
        num_samples=int(data_config["train_samples"]),
        pilot_ratio=pilot_ratio,
        snr_db=snr_db,
        seed=seed,
    )
    test_dataset = _build_dataset(
        data_config,
        system_config,
        num_samples=int(data_config["test_samples"]),
        pilot_ratio=pilot_ratio,
        snr_db=snr_db,
        seed=seed + int(data_config["train_samples"]) + int(data_config["val_samples"]),
    )
    batch_size = int(data_config["batch_size"])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _build_dataset(
    data_config: dict[str, Any],
    system_config: dict[str, Any],
    num_samples: int,
    pilot_ratio: float,
    snr_db: float,
    seed: int,
) -> NoisySyntheticCSIDataset:
    return NoisySyntheticCSIDataset(
        num_samples=num_samples,
        n_tx=int(system_config["n_tx"]),
        n_rx=int(system_config["n_rx"]),
        n_subcarriers=int(system_config["n_subcarriers"]),
        n_paths=int(system_config["n_paths"]),
        pilot_ratio=pilot_ratio,
        snr_db=snr_db,
        seed=seed,
    )


def _train_dncnn(
    model: DnCNNBaseline,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for batch in loader:
            H_sparse = batch["H_sparse"].to(device)
            H_full = batch["H_full"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            H_hat = model(H_sparse, mask)
            loss = loss_fn(H_hat, H_full)
            loss.backward()
            optimizer.step()


def _train_picasso(
    generator: PICASSOGenerator,
    discriminator: PICASSODiscriminator,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> None:
    training_config = config["training"]
    loss_config = config["loss"]
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training_config["learning_rate_g"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(training_config["learning_rate_d"]))
    for _ in range(int(training_config["epochs"])):
        generator.train()
        discriminator.train()
        for batch in loader:
            H_sparse = batch["H_sparse"].to(device)
            H_full = batch["H_full"].to(device)
            mask = batch["mask"].to(device)

            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                H_fake = generator(H_sparse)
            real_logits = discriminator(H_full)
            fake_logits = discriminator(H_fake.detach())
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
            losses["total"].backward()
            optimizer_g.step()


@torch.no_grad()
def _evaluate_estimator(
    loader: DataLoader,
    device: torch.device,
    estimator,
) -> dict[str, float]:
    totals = {
        "samples": 0.0,
        "nmse": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "pilot_consistency_error": 0.0,
        "delay_sparsity_score": 0.0,
    }
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        H_hat = estimator(H_sparse, mask)
        batch_size = float(H_full.shape[0])
        totals["samples"] += batch_size
        totals["nmse"] += float(nmse(H_hat, H_full).detach().cpu()) * batch_size
        totals["mse"] += float(mse(H_hat, H_full).detach().cpu()) * batch_size
        totals["mae"] += float(mae(H_hat, H_full).detach().cpu()) * batch_size
        totals["pilot_consistency_error"] += float(
            pilot_consistency_error(H_hat, H_sparse, mask).detach().cpu()
        ) * batch_size
        totals["delay_sparsity_score"] += float(delay_domain_sparsity_score(H_hat).detach().cpu()) * batch_size
    samples = max(totals["samples"], 1.0)
    return {
        "nmse": totals["nmse"] / samples,
        "mse": totals["mse"] / samples,
        "mae": totals["mae"] / samples,
        "pilot_consistency_error": totals["pilot_consistency_error"] / samples,
        "delay_sparsity_score": totals["delay_sparsity_score"] / samples,
    }


def _make_row(
    config: dict[str, Any],
    method: str,
    pilot_ratio: float,
    snr_db: float,
    metrics: dict[str, float],
    epochs: int,
) -> dict[str, object]:
    return {
        "stage": config["project"]["stage"],
        "method": method,
        "pilot_ratio": f"{pilot_ratio:.4f}",
        "snr_db": f"{snr_db:g}",
        "nmse": f"{metrics['nmse']:.8f}",
        "mse": f"{metrics['mse']:.8f}",
        "mae": f"{metrics['mae']:.8f}",
        "pilot_consistency_error": f"{metrics['pilot_consistency_error']:.8f}",
        "delay_sparsity_score": f"{metrics['delay_sparsity_score']:.8f}",
        "epochs": epochs,
        "num_train_samples": int(config["data"]["train_samples"]),
        "seed": int(config["data"]["seed"]),
    }


def _validate_config(config: dict[str, Any]) -> None:
    data_config = config["data"]
    training_config = config["training"]
    policy = config.get("artifact_policy", {})
    if bool(data_config.get("save_generated_data", False)):
        raise ValueError("Stage 2A must not save generated datasets.")
    if int(data_config["train_samples"]) > 1024:
        raise ValueError("Stage 2A caps train_samples at 1024.")
    if int(data_config["val_samples"]) > 256 or int(data_config["test_samples"]) > 256:
        raise ValueError("Stage 2A caps val/test samples at 256.")
    epochs = int(training_config["epochs"])
    if epochs <= 0 or epochs > 5:
        raise ValueError("Stage 2A requires 1 to 5 training epochs.")
    blocked = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Stage 2A must not save protected artifacts: {enabled}.")
    if not bool(policy.get("save_csv_results", False)):
        raise ValueError("Stage 2A requires save_csv_results: true for the metrics table.")


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
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 2A small formal experiment.")
    parser.add_argument("--config", default="configs/stage2a_small_formal.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
