"""PICASSO generator and discriminator skeletons for smoke testing."""

from __future__ import annotations

import torch
from torch import nn

from picasso_csi.models.conditioning import make_condition_channels


class PICASSOGenerator(nn.Module):
    """Lightweight residual generator for sparse-pilot CSI reconstruction."""

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        base_channels: int = 32,
        num_blocks: int = 3,
        use_condition: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")

        self.use_condition = bool(use_condition)
        if self.use_condition and input_channels == 2:
            input_channels = 4
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(_ResidualBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        H_sparse: torch.Tensor,
        snr_db: float | torch.Tensor | None = None,
        pilot_ratio: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        batched, H_sparse_b = _ensure_batched(H_sparse)
        batch_size, channels, n_rx, n_tx, n_subcarriers = H_sparse_b.shape
        model_input = H_sparse_b.reshape(batch_size, channels, n_rx * n_tx, n_subcarriers)
        if self.use_condition:
            if snr_db is None or pilot_ratio is None:
                raise ValueError("snr_db and pilot_ratio are required when use_condition=True.")
            condition = make_condition_channels(
                batch_size,
                n_rx * n_tx,
                n_subcarriers,
                snr_db,
                pilot_ratio,
                device=model_input.device,
            )
            model_input = torch.cat([model_input, condition], dim=1)
        residual = self.net(model_input)
        residual = residual.reshape(batch_size, channels, n_rx, n_tx, n_subcarriers)
        output = H_sparse_b + residual
        return output if batched else output.squeeze(0)


class PICASSODiscriminator(nn.Module):
    """Small CNN discriminator that emits one real/fake logit per sample."""

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(base_channels * 2, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        batched, H_b = _ensure_batched(H)
        batch_size, channels, n_rx, n_tx, n_subcarriers = H_b.shape
        model_input = H_b.reshape(batch_size, channels, n_rx * n_tx, n_subcarriers)
        features = self.features(model_input).flatten(1)
        logits = self.head(features)
        return logits if batched else logits.squeeze(0)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


def _ensure_batched(tensor: torch.Tensor) -> tuple[bool, torch.Tensor]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("PICASSO models expect torch.Tensor inputs.")
    if tensor.ndim == 5:
        _validate_channels(tensor)
        return True, tensor
    if tensor.ndim == 4:
        _validate_channels(tensor)
        return False, tensor.unsqueeze(0)
    raise ValueError(
        "CSI tensors must have shape (B, 2, n_rx, n_tx, n_subcarriers) "
        "or (2, n_rx, n_tx, n_subcarriers)."
    )


def _validate_channels(tensor: torch.Tensor) -> None:
    if tensor.shape[-4] != 2:
        raise ValueError("CSI tensors must use two real-imaginary channels.")
