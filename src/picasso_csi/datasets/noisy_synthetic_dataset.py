"""Noise-aware synthetic CSI dataset for Stage 2A experiments."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from picasso_csi.simulation import (
    create_pilot_mask,
    generate_mimo_ofdm_channel,
)


SUPPORTED_SNR_DB = (10, 20, 30)


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
    ) -> None:
        self.num_samples = _validate_positive("num_samples", num_samples)
        self.n_tx = _validate_positive("n_tx", n_tx)
        self.n_rx = _validate_positive("n_rx", n_rx)
        self.n_subcarriers = _validate_positive("n_subcarriers", n_subcarriers)
        self.n_paths = _validate_positive("n_paths", n_paths)
        self.pilot_ratio = float(pilot_ratio)
        self.snr_db = _validate_snr(snr_db)
        self.seed = seed
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
                n_paths=self.n_paths,
                seed=self.seed + index,
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
        observed = self._mask_np > 0.0
        signal_values = H_full[observed]
        signal_power = float(np.mean(signal_values**2)) if signal_values.size else 0.0
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise_std = float(np.sqrt(max(noise_power, 0.0)))
        rng = np.random.default_rng(self.seed + 1_000_003 + index)
        noise = np.zeros_like(H_full, dtype=np.float32)
        noise[observed] = rng.normal(loc=0.0, scale=noise_std, size=int(observed.sum()))
        return (H_full * self._mask_np + noise * self._mask_np).astype(np.float32)


def _to_channel_first_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()


def _validate_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _validate_snr(value: int | float) -> float:
    if float(value) not in tuple(float(v) for v in SUPPORTED_SNR_DB):
        raise ValueError(f"snr_db must be one of {SUPPORTED_SNR_DB}, got {value!r}.")
    return float(value)
