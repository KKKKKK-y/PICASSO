"""Minimal MIMO-OFDM channel simulation utilities.

Stage 0 intentionally keeps the simulator lightweight. The generated channel is
an in-memory synthetic frequency-domain multipath channel for shape tests and
interface validation only.
"""

from __future__ import annotations

import numpy as np


def generate_mimo_ofdm_channel(
    n_tx: int,
    n_rx: int,
    n_subcarriers: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic frequency-domain MIMO-OFDM channel.

    Args:
        n_tx: Number of transmit antennas.
        n_rx: Number of receive antennas.
        n_subcarriers: Number of OFDM subcarriers.
        n_paths: Number of multipath components.
        seed: Optional random seed for deterministic tests.

    Returns:
        Real-imaginary channel tensor with shape
        ``(n_rx, n_tx, n_subcarriers, 2)``.
    """

    _validate_positive("n_tx", n_tx)
    _validate_positive("n_rx", n_rx)
    _validate_positive("n_subcarriers", n_subcarriers)
    _validate_positive("n_paths", n_paths)

    rng = np.random.default_rng(seed)
    path_gains = (
        rng.standard_normal((n_rx, n_tx, n_paths))
        + 1j * rng.standard_normal((n_rx, n_tx, n_paths))
    ) / np.sqrt(2.0 * n_paths)
    path_delays = rng.uniform(0.0, 1.0, size=n_paths)
    subcarrier_index = np.arange(n_subcarriers, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * path_delays[:, None] * subcarrier_index[None, :])
    channel_complex = np.einsum("rtp,pk->rtk", path_gains, phase)

    channel = np.stack((channel_complex.real, channel_complex.imag), axis=-1)
    return channel.astype(np.float32)


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
