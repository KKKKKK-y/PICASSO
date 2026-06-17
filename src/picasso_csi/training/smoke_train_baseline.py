"""Stage 1A smoke training for lightweight CSI baselines."""

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
from picasso_csi.evaluation import nmse  # noqa: E402
from picasso_csi.models import CNNBaseline, DnCNNBaseline  # noqa: E402


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    result = run_smoke_training(config)
    print(f"device: {result['device']}")
    print(f"model name: {result['model_name']}")
    print(f"train samples: {result['train_samples']}")
    print(f"val samples: {result['val_samples']}")
    print(f"final val NMSE: {result['final_val_nmse']:.6f}")
    print("smoke training completed")


def run_smoke_training(config: dict[str, Any]) -> dict[str, Any]:
    training_config = config["training"]
    system_config = config["system"]
    data_config = config["data"]
    model_config = config["model"]

    _validate_artifact_policy(config.get("artifact_policy", {}))
    seed = int(training_config.get("seed", 42))
    _set_seed(seed)

    use_cuda = bool(training_config.get("use_cuda", True))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    train_dataset = SyntheticCSIDataset(
        num_samples=int(data_config["train_samples"]),
        n_tx=int(system_config["n_tx"]),
        n_rx=int(system_config["n_rx"]),
        n_subcarriers=int(system_config["n_subcarriers"]),
        n_paths=int(system_config["n_paths"]),
        pilot_ratio=float(system_config["pilot_ratio"]),
        seed=seed,
    )
    val_dataset = SyntheticCSIDataset(
        num_samples=int(data_config["val_samples"]),
        n_tx=int(system_config["n_tx"]),
        n_rx=int(system_config["n_rx"]),
        n_subcarriers=int(system_config["n_subcarriers"]),
        n_paths=int(system_config["n_paths"]),
        pilot_ratio=float(system_config["pilot_ratio"]),
        seed=seed + int(data_config["train_samples"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model_name = str(model_config.get("name", "cnn_baseline"))
    model = _build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))
    epochs = int(training_config["epochs"])

    final_val_nmse = float("nan")
    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        final_val_nmse = _evaluate_nmse(model, val_loader, device)
        print(f"epoch {epoch}/{epochs} - train loss: {train_loss:.6f} - val NMSE: {final_val_nmse:.6f}")

    return {
        "device": str(device),
        "model_name": model_name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_val_nmse": final_val_nmse,
    }


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_fn = torch.nn.MSELoss()
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        H_hat = model(H_sparse, mask)
        loss = loss_fn(H_hat, H_full)
        loss.backward()
        optimizer.step()
        batch_size = H_full.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _evaluate_nmse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_nmse = 0.0
    total_samples = 0
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        H_hat = model(H_sparse, mask)
        batch_nmse = nmse(H_hat, H_full)
        batch_size = H_full.shape[0]
        total_nmse += float(batch_nmse.cpu()) * batch_size
        total_samples += batch_size
    return total_nmse / max(total_samples, 1)


def _build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(model_config.get("name", "cnn_baseline")).lower()
    kwargs = {
        "input_channels": int(model_config.get("input_channels", 4)),
        "output_channels": int(model_config.get("output_channels", 2)),
        "hidden_channels": int(model_config.get("hidden_channels", 32)),
    }
    if model_name == "cnn_baseline":
        return CNNBaseline(**kwargs)
    if model_name == "dncnn_baseline":
        return DnCNNBaseline(**kwargs)
    raise ValueError(f"Unsupported smoke baseline model {model_name!r}.")


def _validate_artifact_policy(policy: dict[str, Any]) -> None:
    blocked = [
        "save_checkpoints",
        "save_numpy_data",
        "save_outputs",
    ]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Stage 1A smoke training must not save artifacts: {enabled}.")


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
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 1A smoke training.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Path to smoke YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
