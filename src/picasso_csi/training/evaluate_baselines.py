"""Stage 1B smoke-level baseline evaluation."""

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
from picasso_csi.evaluation import (  # noqa: E402
    delay_domain_sparsity_score,
    mae,
    mse,
    nmse,
    pilot_consistency_error,
)
from picasso_csi.models import (  # noqa: E402
    CNNBaseline,
    DnCNNBaseline,
    lmmse_like_baseline,
    ls_baseline,
    omp_like_baseline,
)

MetricDict = dict[str, float]


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    results = run_evaluation(config)
    print(f"device: {results['device']}")
    print(f"samples: {results['samples']}")
    print("baseline results:")
    for name, metrics in results["baselines"].items():
        metric_text = ", ".join(f"{key}: {value:.6f}" for key, value in metrics.items())
        print(f"- {name}: {metric_text}")
    print("stage1b baseline evaluation completed")


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    _validate_limits(config)
    _validate_artifact_policy(config.get("artifact_policy", {}))
    data_config = config["data"]
    system_config = config["system"]
    evaluation_config = config["evaluation"]
    seed = int(data_config.get("seed", 42))
    _set_seed(seed)

    requested_device = str(evaluation_config.get("device", "cuda")).lower()
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = SyntheticCSIDataset(
        num_samples=int(data_config["num_samples"]),
        n_tx=int(system_config["n_tx"]),
        n_rx=int(system_config["n_rx"]),
        n_subcarriers=int(system_config["n_subcarriers"]),
        n_paths=int(system_config["n_paths"]),
        pilot_ratio=float(system_config["pilot_ratio"]),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    results: dict[str, MetricDict] = {}
    if bool(evaluation_config.get("run_classical", True)):
        results["LS"] = _evaluate_estimator(loader, device, lambda H_sparse, mask: ls_baseline(H_sparse, mask))
        results["LMMSE-like"] = _evaluate_estimator(
            loader,
            device,
            lambda H_sparse, mask: lmmse_like_baseline(H_sparse, mask, smoothing=True),
        )
        results["OMP-like"] = _evaluate_estimator(
            loader,
            device,
            lambda H_sparse, mask: omp_like_baseline(
                H_sparse,
                mask,
                n_paths=int(system_config["n_paths"]),
            ),
        )

    if bool(evaluation_config.get("run_neural_smoke", True)):
        neural_epochs = int(evaluation_config["neural_epochs"])
        results["CNN smoke"] = _train_and_evaluate_neural(
            CNNBaseline(),
            loader,
            device,
            epochs=neural_epochs,
        )
        results["DnCNN smoke"] = _train_and_evaluate_neural(
            DnCNNBaseline(),
            loader,
            device,
            epochs=neural_epochs,
        )

    return {
        "device": str(device),
        "samples": len(dataset),
        "baselines": results,
    }


@torch.no_grad()
def _evaluate_estimator(
    loader: DataLoader,
    device: torch.device,
    estimator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> MetricDict:
    totals = _empty_totals()
    for batch in loader:
        H_sparse = batch["H_sparse"].to(device)
        H_full = batch["H_full"].to(device)
        mask = batch["mask"].to(device)
        H_hat = estimator(H_sparse, mask)
        _accumulate_metrics(totals, H_hat, H_full, H_sparse, mask)
    return _finalize_metrics(totals)


def _train_and_evaluate_neural(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> MetricDict:
    model.to(device)
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

    model.eval()
    return _evaluate_estimator(loader, device, lambda H_sparse, mask: model(H_sparse, mask))


def _empty_totals() -> dict[str, float]:
    return {
        "samples": 0.0,
        "NMSE": 0.0,
        "MSE": 0.0,
        "MAE": 0.0,
        "pilot consistency": 0.0,
        "delay sparsity": 0.0,
    }


def _accumulate_metrics(
    totals: dict[str, float],
    H_hat: torch.Tensor,
    H_full: torch.Tensor,
    H_sparse: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    batch_size = float(H_full.shape[0] if H_full.ndim == 5 else 1)
    totals["samples"] += batch_size
    totals["NMSE"] += float(nmse(H_hat, H_full).detach().cpu()) * batch_size
    totals["MSE"] += float(mse(H_hat, H_full).detach().cpu()) * batch_size
    totals["MAE"] += float(mae(H_hat, H_full).detach().cpu()) * batch_size
    totals["pilot consistency"] += float(
        pilot_consistency_error(H_hat, H_sparse, mask).detach().cpu()
    ) * batch_size
    totals["delay sparsity"] += float(
        delay_domain_sparsity_score(H_hat).detach().cpu()
    ) * batch_size


def _finalize_metrics(totals: dict[str, float]) -> MetricDict:
    samples = max(totals["samples"], 1.0)
    return {
        "NMSE": totals["NMSE"] / samples,
        "MSE": totals["MSE"] / samples,
        "MAE": totals["MAE"] / samples,
        "pilot consistency": totals["pilot consistency"] / samples,
        "delay sparsity": totals["delay sparsity"] / samples,
    }


def _validate_limits(config: dict[str, Any]) -> None:
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    if bool(data_config.get("save_generated_data", False)):
        raise ValueError("Stage 1B smoke evaluation must not save generated data.")
    if int(data_config["num_samples"]) > 256:
        raise ValueError("Stage 1B smoke evaluation caps num_samples at 256.")
    neural_epochs = int(evaluation_config["neural_epochs"])
    if neural_epochs < 0:
        raise ValueError("neural_epochs must be non-negative.")
    if neural_epochs > 2:
        raise ValueError("Stage 1B smoke evaluation caps neural_epochs at 2.")


def _validate_artifact_policy(policy: dict[str, Any]) -> None:
    blocked = ["save_checkpoints", "save_outputs", "save_numpy_arrays"]
    enabled = [key for key in blocked if bool(policy.get(key, False))]
    if enabled:
        raise ValueError(f"Stage 1B evaluation must not save artifacts: {enabled}.")


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
    parser = argparse.ArgumentParser(description="Run PICASSO Stage 1B baseline evaluation.")
    parser.add_argument("--config", default="configs/stage1b_baselines.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
