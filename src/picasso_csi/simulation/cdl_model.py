"""Lightweight 3GPP-CDL-inspired channel simulation utilities.

This module is not a full 3GPP TR 38.901 implementation. It keeps the project
self-contained while adding CDL-style clustered multipath, angular/spatial
correlation, delay spread variation, and velocity-dependent Doppler phase.
"""

from __future__ import annotations

import numpy as np


CDL_PROFILES = {
    "CDL-A": {
        "delays": np.array([0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750]),
        "powers_db": np.array([-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5]),
        "angular_spread_deg": 35.0,
    },
    "CDL-B": {
        "delays": np.array([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055]),
        "powers_db": np.array([0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2]),
        "angular_spread_deg": 25.0,
    },
    "CDL-C": {
        "delays": np.array([0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560]),
        "powers_db": np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9]),
        "angular_spread_deg": 15.0,
    },
}


def generate_cdl_mimo_ofdm_channel(
    n_tx: int,
    n_rx: int,
    n_subcarriers: int,
    profile: str = "CDL-A",
    velocity_kmh: float = 0.0,
    carrier_frequency_hz: float = 3.5e9,
    sample_time_s: float = 1.0e-3,
    time_index: int = 0,
    delay_spread_scale: float = 1.0,
    seed: int | None = None,
    normalize_channel: bool = True,
) -> np.ndarray:
    """Generate a frequency-domain clustered MIMO-OFDM channel.

    Returns a real-imaginary array with shape ``(n_rx, n_tx, n_subcarriers, 2)``.
    """

    if profile not in CDL_PROFILES:
        raise ValueError(f"Unsupported CDL profile {profile!r}.")
    rng = np.random.default_rng(seed)
    params = CDL_PROFILES[profile]
    delays = params["delays"].astype(np.float64) * float(delay_spread_scale)
    powers = 10.0 ** (params["powers_db"].astype(np.float64) / 10.0)
    powers = powers / np.sum(powers)
    n_clusters = delays.size

    mean_aoa = rng.uniform(-np.pi / 3.0, np.pi / 3.0, size=n_clusters)
    mean_aod = rng.uniform(-np.pi / 3.0, np.pi / 3.0, size=n_clusters)
    spread = np.deg2rad(float(params["angular_spread_deg"]))
    aoa = mean_aoa + rng.normal(0.0, spread / 3.0, size=n_clusters)
    aod = mean_aod + rng.normal(0.0, spread / 3.0, size=n_clusters)
    gains = (
        rng.standard_normal(n_clusters) + 1j * rng.standard_normal(n_clusters)
    ) * np.sqrt(powers / 2.0)

    speed_mps = float(velocity_kmh) / 3.6
    wavelength = 3.0e8 / float(carrier_frequency_hz)
    max_doppler = speed_mps / wavelength
    doppler_hz = max_doppler * np.cos(aoa)
    time_s = float(time_index) * float(sample_time_s)
    doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * time_s)

    rx_response = _ula_response(n_rx, aoa)
    tx_response = _ula_response(n_tx, aod)
    subcarrier_index = np.arange(n_subcarriers, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * delays[:, None] * subcarrier_index[None, :])

    channel = np.zeros((n_rx, n_tx, n_subcarriers), dtype=np.complex128)
    for cluster in range(n_clusters):
        spatial = rx_response[:, cluster][:, None] * np.conj(tx_response[:, cluster][None, :])
        channel += gains[cluster] * doppler_phase[cluster] * spatial[:, :, None] * phase[cluster][None, None, :]

    if normalize_channel:
        rms = np.sqrt(np.mean(np.abs(channel) ** 2))
        if rms > 0:
            channel = channel / rms
    return np.stack([channel.real, channel.imag], axis=-1).astype(np.float32)


def _ula_response(n_antennas: int, angle_rad: np.ndarray) -> np.ndarray:
    antenna_index = np.arange(n_antennas, dtype=np.float64)[:, None]
    return np.exp(1j * np.pi * antenna_index * np.sin(angle_rad)[None, :]) / np.sqrt(float(n_antennas))
