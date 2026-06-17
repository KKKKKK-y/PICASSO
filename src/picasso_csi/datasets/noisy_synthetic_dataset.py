"""Noise-aware synthetic CSI dataset for Stage 2 experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from picasso_csi.simulation import (
    create_pilot_mask,
    generate_mimo_ofdm_channel,
)


class NoisySyntheticCSIDataset(Dataset):
    """Generate clean CSI labels and noisy sparse pilot observations in memory."""

    def __init__(
        self,
        num_samples: int,
        n_tx: int = 4,
        n_rx: int = 4,
        n_subcarriers: int = 64,
        n_paths: int = 6,
        pilot_ratio: float = 0.25,
        snr_db: int | float = 20,
        seed: int = 42,
        random_n_paths: bool = False,
        min_paths: int = 3,
        max_paths: int = 10,
        delay_spread: float | str = 1.0,
        gain_distribution: str = "complex_gaussian",
        normalize_channel: bool = False,
        pilot_noise_only: bool = True,
    ) -> None:
        self.num_samples = _validate_positive("num_samples", num_samples)
        self.n_tx = _validate_positive("n_tx", n_tx)
        self.n_rx = _validate_positive("n_rx", n_rx)
        self.n_subcarriers = _validate_positive("n_subcarriers", n_subcarriers)
        self.n_paths = _validate_positive("n_paths", n_paths)
        self.pilot_ratio = float(pilot_ratio)
        self.snr_db = _validate_snr(snr_db)
        self.seed = seed
        self.random_n_paths = bool(random_n_paths)
        self.min_paths = _validate_positive("min_paths", min_paths)
        self.max_paths = _validate_positive("max_paths", max_paths)
        if self.min_paths > self.max_paths:
            raise ValueError("min_paths must be <= max_paths.")
        self.delay_spread = delay_spread
        self.gain_distribution = gain_distribution
        self.normalize_channel = bool(normalize_channel)
        self.pilot_noise_only = bool(pilot_noise_only)
        self._mask_np = create_pilot_mask(
            n_rx=self.n_rx,
            n_tx=self.n_tx,
            n_subcarriers=self.n_subcarriers,
            pilot_ratio=self.pilot_ratio,
        )
        self._H_full, self._H_sparse, self._mask = self._build_tensors()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} is out of range for {self.num_samples} samples.")
        return {
            "H_full": self._H_full[index],
            "H_sparse": self._H_sparse[index],
            "mask": self._mask[index],
            "snr_db": torch.tensor(float(self.snr_db), dtype=torch.float32),
            "pilot_ratio": torch.tensor(float(self.pilot_ratio), dtype=torch.float32),
        }

    def _build_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_full_list = []
        H_sparse_list = []
        mask_list = []
        for index in range(self.num_samples):
            H_full = generate_mimo_ofdm_channel(
                n_tx=self.n_tx,
                n_rx=self.n_rx,
                n_subcarriers=self.n_subcarriers,
                n_paths=self._sample_n_paths(index),
                seed=self.seed + index,
                delay_spread=self.delay_spread,
                gain_distribution=self.gain_distribution,
                normalize_channel=self.normalize_channel,
            )
            H_sparse = self._apply_noisy_pilot_observation(H_full, index)
            H_full_list.append(_to_channel_first_tensor(H_full))
            H_sparse_list.append(_to_channel_first_tensor(H_sparse))
            mask_list.append(_to_channel_first_tensor(self._mask_np))
        return (
            torch.stack(H_full_list, dim=0),
            torch.stack(H_sparse_list, dim=0),
            torch.stack(mask_list, dim=0),
        )

    def _apply_noisy_pilot_observation(self, H_full: np.ndarray, index: int) -> np.ndarray:
        observed = self._mask_np > 0.0 if self.pilot_noise_only else np.ones_like(self._mask_np, dtype=bool)
        pilot_observed = self._mask_np > 0.0
        signal_values = H_full[pilot_observed]
        signal_power = float(np.mean(signal_values**2)) if signal_values.size else float(np.mean(H_full**2))
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise_std = float(np.sqrt(max(noise_power, 0.0)))
        rng = np.random.default_rng(self.seed + 1_000_003 + index)
        noise = np.zeros_like(H_full, dtype=np.float32)
        noise[observed] = rng.normal(loc=0.0, scale=noise_std, size=int(observed.sum()))
        return ((H_full + noise) * self._mask_np).astype(np.float32)

    def _sample_n_paths(self, index: int) -> int:
        if not self.random_n_paths:
            return self.n_paths
        rng = np.random.default_rng(self.seed + 2_000_033 + index)
        return int(rng.integers(self.min_paths, self.max_paths + 1))


def _to_channel_first_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()


def _validate_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _validate_snr(value: int | float) -> float:
    snr = float(value)
    if not np.isfinite(snr):
        raise ValueError(f"snr_db must be finite, got {value!r}.")
    return snr
