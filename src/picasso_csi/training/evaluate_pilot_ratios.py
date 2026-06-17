"""Smoke-level pilot-ratio evaluation for LS, DnCNN, and PICASSO generator."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.datasets import SyntheticCSIDataset  # noqa: E402
from picasso_csi.evaluation import nmse, pilot_consistency_error  # noqa: E402
from picasso_csi.losses import picasso_generator_loss  # noqa: E402
from picasso_csi.models import DnCNNBaseline, PICASSOGenerator, ls_baseline  # noqa: E402


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_pilot_ratio_evaluation(config)
    print(f"device: {result['device']}")
    print("pilot ratio results:")
    for row in result["rows"]:
        print(
            f"- ratio {row['pilot_ratio']:.4f}: "
            f"LS NMSE: {row['LS NMSE']:.6f}, "
            f"DnCNN NMSE: {row['DnCNN NMSE']:.6f}, "
            f"PICASSO NMSE: {row['PICASSO NMSE']:.6f}, "
            f"PICASSO pilot consistency: {row['PICASSO pilot consistency']:.6f}"
        )
    print("pilot ratio smoke evaluation completed")


def run_pilot_ratio_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    _validate_smoke_config(config)
    data_config = config["data"]
    system_config = config["system"]
    training_config = config["training"]
    seed = int(data_config.get("seed", 42))
    _set_seed(seed)
    device = _resolve_device(str(training_config.get("device", "cuda")))
    rows = []
    for index, ratio in enumerate(system_config["pilot_ratios_eval"]):
        ratio = float(ratio)
        ratio_seed = seed + index * 1000
        train_loader, val_loader = _build_loaders(data_config, system_config, ratio, ratio_seed)

        ls_metrics = _evaluate_estimator(val_loader, device, lambda H_sparse, mask: ls_baseline(H_sparse, mask))

        dncnn = DnCNNBaseline().to(device)
        _train_dncnn(dncnn, train_loader, device, epochs=int(training_config["smoke_epochs"]))
        dncnn_metrics = _evaluate_estimator(val_loader, device, lambda H_sparse, mask: dncnn(H_sparse, mask))

        generator = PICASSOGenerator(**config["model"]["generator"]).to(device)
        _train_generator(generator, train_loader, device, config["loss"], epochs=int(training_config["smoke_epochs"]))
        picasso_metrics = _evaluate_estimator(val_loader, device, lambda H_sparse, mask: generator(H_sparse))

        rows.append(
            {
                "pilot_ratio": ratio,
                "LS NMSE": ls_metrics["NMSE"],
                "DnCNN NMSE": dncnn_metrics["NMSE"],
                "PICASSO NMSE": picasso_metrics["NMSE"],
                "PICASSO pilot consistency": picasso_metrics["pilot consistency"],
            }
        )
    return {"device": str(device), "rows": rows}


def _build_loaders(
    data_config: dict[str, Any],
    system_config: dict[str, Any],
    pilot_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    num_samples = int(data_config["num_samples"])
    val_samples = max(1, min(64, num_samples // 4))
    train_dataset = _build_dataset(data_config, system_config, pilot_ratio, num_samples, seed)
    val_dataset = _build_dataset(data_config, system_config, pilot_ratio, val_samples, seed + num_samples)
    batch_size = int(data_config["batch_size"])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _build_dataset(
    data_config: dict[str, Any],
    system_config: dict[str, Any],
    pilot_ratio: float,
    num_samples: int,
    seed: int,
) -> SyntheticCSIDataset:
    return SyntheticCSIDataset(
        num_samples=num_samples,
        n_tx=int(system_config["n_tx"]),
        n_rx=int(system_config["n_rx"]),
        n_subcarriers=int(system_config["n_subcarriers"]),
        n_paths=int(system_config["n_paths"]),
        pilot_ratio=pilot_ratio,
        seed=seed,
    )


def _train_dncnn(
    model: DnCNNBaseline,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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


def _train_generator(
    generator: PICASSOGenerator,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict[str, Any],
    epochs: int,
) -> None:
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    for _ in range(epochs):
        generator.train()
        for batch in loader:
            H_sparse = batch["H_sparse"].to(device)
            H_full = batch["H_full"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            H_hat = generator(H_sparse)
            losses = picasso_generator_loss(
                H_hat,
                H_full,
                H_sparse,
                mask,
                fake_logits=None,
                lambda_rec=float(loss_config["lambda_rec"]),
                lambda_adv=0.0,
                lambda_pilot=float(loss_config["lambda_pilot"]),
                lambda_smooth=float(loss_config["lambda_smooth"]),
                lambda_sparse=float(loss_config["lambda_sparse"]),
            )
            losses["total"].backward()
            optimizer.step()


@torch.no_grad()
def _evaluate_estimator(
    loader: DataLoader,
    device: torch.device,
    estimator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, float]:
    total_nmse = 0.0
    total_pilot = 0.0
    total_samples = 0
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        H_hat = estimator(H_sparse, mask)
        batch_size = H_full.shape[0]
        total_nmse += float(nmse(H_hat, H_full).detach().cpu()) * batch_size
        total_pilot += float(pilot_consistency_error(H_hat, H_sparse, mask).detach().cpu()) * batch_size
        total_samples += batch_size
    return {
        "NMSE": total_nmse / max(total_samples, 1),
        "pilot consistency": total_pilot / max(total_samples, 1),
    }


def _validate_smoke_config(config: dict[str, Any]) -> None:
    data_config = config["data"]
    training_config = config["training"]
    policy = config.get("artifact_policy", {})
    if bool(data_config.get("save_generated_data", False)):
        raise ValueError("Pilot ratio smoke evaluation must not save generated data.")
    if int(data_config["num_samples"]) > 256:
        raise ValueError("Pilot ratio smoke evaluation caps num_samples at 256.")
    epochs = int(training_config["smoke_epochs"])
    if epochs <= 0 or epochs > 2:
        raise ValueError("Pilot ratio smoke evaluation requires 1 or 2 epochs.")
    blocked = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Pilot ratio smoke evaluation must not save artifacts: {enabled}.")


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
    parser = argparse.ArgumentParser(description="Run pilot-ratio smoke evaluation.")
    parser.add_argument("--config", default="configs/stage1c_picasso_smoke.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
