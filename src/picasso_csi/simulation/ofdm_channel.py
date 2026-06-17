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
    delay_spread: float | str = 1.0,
    gain_distribution: str = "complex_gaussian",
    normalize_channel: bool = False,
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
    if gain_distribution != "complex_gaussian":
        raise ValueError(f"Unsupported gain_distribution {gain_distribution!r}.")

    spread = _resolve_delay_spread(delay_spread, rng)
    path_gains = (
        rng.standard_normal((n_rx, n_tx, n_paths))
        + 1j * rng.standard_normal((n_rx, n_tx, n_paths))
    ) / np.sqrt(2.0 * n_paths)
    path_delays = rng.uniform(0.0, spread, size=n_paths)
    subcarrier_index = np.arange(n_subcarriers, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * path_delays[:, None] * subcarrier_index[None, :])
    channel_complex = np.einsum("rtp,pk->rtk", path_gains, phase)

    if normalize_channel:
        rms = np.sqrt(np.mean(np.abs(channel_complex) ** 2))
        if rms > 0:
            channel_complex = channel_complex / rms

    channel = np.stack((channel_complex.real, channel_complex.imag), axis=-1)
    return channel.astype(np.float32)


def _resolve_delay_spread(delay_spread: float | str, rng: np.random.Generator) -> float:
    if isinstance(delay_spread, str):
        if delay_spread.lower() != "random":
            raise ValueError(f"Unsupported delay_spread {delay_spread!r}.")
        return float(rng.uniform(0.25, 1.5))
    spread = float(delay_spread)
    if spread <= 0:
        raise ValueError(f"delay_spread must be positive, got {delay_spread!r}.")
    return spread


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
