"""CDL-inspired CSI dataset with mobility and realistic pilot patterns."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from picasso_csi.simulation.cdl_model import generate_cdl_mimo_ofdm_channel
from picasso_csi.simulation.pilot_mask import create_pilot_mask


PROFILE_TO_ID = {"CDL-A": 0, "CDL-B": 1, "CDL-C": 2}


class CDLChannelDataset(Dataset):
    """Generate CDL-style clean CSI labels and noisy sparse pilot observations."""

    def __init__(
        self,
        num_samples: int,
        n_tx: int = 4,
        n_rx: int = 4,
        n_subcarriers: int = 64,
        profile: str = "CDL-A",
        pilot_ratio: float = 0.25,
        pilot_pattern: str = "comb",
        snr_db: float = 20.0,
        velocity_kmh: float = 0.0,
        seed: int = 42,
        delay_spread_scale: float | str = "random",
        pilot_contamination_std: float = 0.0,
    ) -> None:
        self.num_samples = _positive_int("num_samples", num_samples)
        self.n_tx = _positive_int("n_tx", n_tx)
        self.n_rx = _positive_int("n_rx", n_rx)
        self.n_subcarriers = _positive_int("n_subcarriers", n_subcarriers)
        if profile not in PROFILE_TO_ID:
            raise ValueError(f"Unsupported CDL profile {profile!r}.")
        self.profile = profile
        self.pilot_ratio = float(pilot_ratio)
        self.pilot_pattern = pilot_pattern
        self.snr_db = float(snr_db)
        self.velocity_kmh = float(velocity_kmh)
        self.seed = int(seed)
        self.delay_spread_scale = delay_spread_scale
        self.pilot_contamination_std = float(pilot_contamination_std)
        self.mask_np = create_pilot_mask(
            self.n_rx,
            self.n_tx,
            self.n_subcarriers,
            self.pilot_ratio,
            pattern=self.pilot_pattern,
        )
        self._H_full, self._H_prev, self._H_sparse, self._mask = self._build()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "H_full": self._H_full[index],
            "H_prev": self._H_prev[index],
            "H_sparse": self._H_sparse[index],
            "mask": self._mask[index],
            "snr_db": torch.tensor(float(self.snr_db), dtype=torch.float32),
            "pilot_ratio": torch.tensor(float(self.pilot_ratio), dtype=torch.float32),
            "velocity_kmh": torch.tensor(float(self.velocity_kmh), dtype=torch.float32),
            "profile_id": torch.tensor(PROFILE_TO_ID[self.profile], dtype=torch.long),
        }

    def _build(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full, prev, sparse, masks = [], [], [], []
        for index in range(self.num_samples):
            scale = self._sample_delay_scale(index)
            seed = self.seed + index
            H_prev = generate_cdl_mimo_ofdm_channel(
                self.n_tx,
                self.n_rx,
                self.n_subcarriers,
                profile=self.profile,
                velocity_kmh=self.velocity_kmh,
                time_index=0,
                delay_spread_scale=scale,
                seed=seed,
            )
            H_full = generate_cdl_mimo_ofdm_channel(
                self.n_tx,
                self.n_rx,
                self.n_subcarriers,
                profile=self.profile,
                velocity_kmh=self.velocity_kmh,
                time_index=1,
                delay_spread_scale=scale,
                seed=seed,
            )
            H_sparse = self._observe_pilots(H_full, index)
            full.append(_to_channel_first(H_full))
            prev.append(_to_channel_first(H_prev))
            sparse.append(_to_channel_first(H_sparse))
            masks.append(_to_channel_first(self.mask_np))
        return torch.stack(full), torch.stack(prev), torch.stack(sparse), torch.stack(masks)

    def _sample_delay_scale(self, index: int) -> float:
        if self.delay_spread_scale != "random":
            return float(self.delay_spread_scale)
        rng = np.random.default_rng(self.seed + 75_031 + index)
        return float(rng.uniform(0.6, 1.8))

    def _observe_pilots(self, H_full: np.ndarray, index: int) -> np.ndarray:
        observed = self.mask_np > 0.0
        signal_values = H_full[observed]
        signal_power = float(np.mean(signal_values**2)) if signal_values.size else float(np.mean(H_full**2))
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise_std = float(np.sqrt(max(noise_power, 0.0)))
        rng = np.random.default_rng(self.seed + 990_001 + index)
        noise = np.zeros_like(H_full, dtype=np.float32)
        noise[observed] = rng.normal(0.0, noise_std, size=int(observed.sum()))
        if self.pilot_contamination_std > 0:
            contamination = rng.normal(0.0, self.pilot_contamination_std, size=H_full.shape).astype(np.float32)
            noise += contamination * self.mask_np
        return ((H_full + noise) * self.mask_np).astype(np.float32)


def _to_channel_first(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value
