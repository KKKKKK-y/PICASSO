"""Synthetic in-memory CSI datasets for Stage 1A smoke tests."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from picasso_csi.simulation import (
    apply_pilot_mask,
    create_pilot_mask,
    generate_mimo_ofdm_channel,
)


class SyntheticCSIDataset(Dataset):
    """Generate small MIMO-OFDM CSI samples on demand.

    Samples are deterministic for a given ``seed`` and index. No generated CSI
    tensors are written to disk.
    """

    def __init__(
        self,
        num_samples: int,
        n_tx: int = 4,
        n_rx: int = 4,
        n_subcarriers: int = 64,
        n_paths: int = 6,
        pilot_ratio: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.num_samples = _validate_positive("num_samples", num_samples)
        self.n_tx = _validate_positive("n_tx", n_tx)
        self.n_rx = _validate_positive("n_rx", n_rx)
        self.n_subcarriers = _validate_positive("n_subcarriers", n_subcarriers)
        self.n_paths = _validate_positive("n_paths", n_paths)
        self.pilot_ratio = pilot_ratio
        self.seed = seed
        self._mask = create_pilot_mask(
            n_rx=self.n_rx,
            n_tx=self.n_tx,
            n_subcarriers=self.n_subcarriers,
            pilot_ratio=self.pilot_ratio,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} is out of range for {self.num_samples} samples.")

        H_full = generate_mimo_ofdm_channel(
            n_tx=self.n_tx,
            n_rx=self.n_rx,
            n_subcarriers=self.n_subcarriers,
            n_paths=self.n_paths,
            seed=self.seed + index,
        )
        H_sparse = apply_pilot_mask(H_full, self._mask)

        return {
            "H_sparse": _to_channel_first_tensor(H_sparse),
            "H_full": _to_channel_first_tensor(H_full),
            "mask": _to_channel_first_tensor(self._mask),
        }

    def metadata(self) -> dict[str, Any]:
        """Return the dataset settings used to generate samples."""

        return {
            "num_samples": self.num_samples,
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "n_paths": self.n_paths,
            "pilot_ratio": self.pilot_ratio,
            "seed": self.seed,
        }


def _to_channel_first_tensor(array: Any) -> torch.Tensor:
    return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()


def _validate_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value
