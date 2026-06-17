"""Condition channel helpers for SNR and pilot-ratio aware models."""

from __future__ import annotations

import torch


def normalize_snr(
    snr_db: float | torch.Tensor,
    min_snr: float = 0.0,
    max_snr: float = 40.0,
) -> torch.Tensor:
    """Normalize SNR in dB to roughly [0, 1]."""

    value = _as_float_tensor(snr_db)
    denom = max(max_snr - min_snr, torch.finfo(value.dtype).eps)
    return torch.clamp((value - min_snr) / denom, 0.0, 1.0)


def normalize_pilot_ratio(pilot_ratio: float | torch.Tensor) -> torch.Tensor:
    """Clamp pilot ratio to [0, 1] for condition channels."""

    return torch.clamp(_as_float_tensor(pilot_ratio), 0.0, 1.0)


def make_condition_channels(
    batch_size: int,
    height: int,
    width: int,
    snr_db: float | torch.Tensor,
    pilot_ratio: float | torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create two spatially constant condition channels.

    Returns a tensor with shape ``(batch_size, 2, height, width)`` where channel
    0 stores normalized SNR and channel 1 stores normalized pilot ratio.
    """

    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch_size, height, and width must be positive.")

    target_device = torch.device(device) if device is not None else None
    snr = normalize_snr(snr_db).flatten()
    ratio = normalize_pilot_ratio(pilot_ratio).flatten()
    if target_device is not None:
        snr = snr.to(target_device)
        ratio = ratio.to(target_device)

    snr = _match_batch(snr, batch_size)
    ratio = _match_batch(ratio, batch_size)
    stacked = torch.stack([snr, ratio], dim=1)
    return stacked[:, :, None, None].expand(batch_size, 2, height, width).contiguous()


def _as_float_tensor(value: float | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32)
    return torch.tensor(float(value), dtype=torch.float32)


def _match_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.numel() == 1:
        return value.expand(batch_size)
    if value.numel() != batch_size:
        raise ValueError(f"Condition value has {value.numel()} elements for batch size {batch_size}.")
    return value
