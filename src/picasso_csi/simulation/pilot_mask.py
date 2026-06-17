"""Pilot mask utilities for sparse-pilot CSI reconstruction."""

from __future__ import annotations

import numpy as np


SUPPORTED_PILOT_RATIOS = (0.5, 0.25, 0.125, 0.0625)


def create_pilot_mask(
    n_rx: int,
    n_tx: int,
    n_subcarriers: int,
    pilot_ratio: float,
    pattern: str = "uniform",
) -> np.ndarray:
    """Create a pilot mask compatible with real-imaginary CSI tensors.

    Args:
        n_rx: Number of receive antennas.
        n_tx: Number of transmit antennas.
        n_subcarriers: Number of OFDM subcarriers.
        pilot_ratio: Fraction of subcarriers observed as pilots.
        pattern: Pilot pattern. Stage 0 supports only ``"uniform"``.

    Returns:
        Float mask with shape ``(n_rx, n_tx, n_subcarriers, 2)``.
    """

    _validate_positive("n_rx", n_rx)
    _validate_positive("n_tx", n_tx)
    _validate_positive("n_subcarriers", n_subcarriers)
    if pattern != "uniform":
        raise ValueError(f"Unsupported pilot pattern {pattern!r}; expected 'uniform'.")
    if pilot_ratio not in SUPPORTED_PILOT_RATIOS:
        raise ValueError(
            f"Unsupported pilot_ratio {pilot_ratio!r}; expected one of {SUPPORTED_PILOT_RATIOS}."
        )

    step = int(round(1.0 / pilot_ratio))
    pilot_indices = np.arange(0, n_subcarriers, step)
    if pilot_indices.size == 0:
        pilot_indices = np.array([0], dtype=np.int64)

    mask = np.zeros((n_rx, n_tx, n_subcarriers, 2), dtype=np.float32)
    mask[:, :, pilot_indices, :] = 1.0
    return mask


def apply_pilot_mask(H: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a pilot mask to a CSI tensor."""

    if H.shape != mask.shape:
        raise ValueError(f"H and mask must have the same shape, got {H.shape} and {mask.shape}.")
    return H * mask


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
