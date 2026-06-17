"""Minimal baseline models for Stage 1A smoke training."""

from __future__ import annotations

import torch
from torch import nn


class LSInterpolationBaseline(nn.Module):
    """Nearest-pilot interpolation baseline along the subcarrier axis."""

    def forward(self, H_sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_csi_pair(H_sparse, mask)
        batched, H_sparse_b = _ensure_batched(H_sparse)
        _, mask_b = _ensure_batched(mask)

        n_subcarriers = H_sparse_b.shape[-1]
        values = H_sparse_b.reshape(-1, n_subcarriers)
        observed = mask_b.reshape(-1, n_subcarriers) > 0
        interpolated = torch.stack(
            [
                _nearest_interpolate(values_row, observed_row)
                for values_row, observed_row in zip(values, observed)
            ],
            dim=0,
        )
        output = interpolated.reshape_as(H_sparse_b)
        return output if batched else output.squeeze(0)


class CNNBaseline(nn.Module):
    """Small Conv2d baseline over antenna-pair and frequency grids."""

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 2,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, H_sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_csi_pair(H_sparse, mask)
        batched, H_sparse_b = _ensure_batched(H_sparse)
        _, mask_b = _ensure_batched(mask)
        model_input, shape = _flatten_antenna_grid(H_sparse_b, mask_b)
        output = self.net(model_input)
        output = _restore_antenna_grid(output, shape)
        return output if batched else output.squeeze(0)


class DnCNNBaseline(nn.Module):
    """Residual CNN baseline with ``H_hat = H_sparse + residual``."""

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 2,
        hidden_channels: int = 32,
        depth: int = 5,
    ) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("depth must be at least 3.")

        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, H_sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_csi_pair(H_sparse, mask)
        batched, H_sparse_b = _ensure_batched(H_sparse)
        _, mask_b = _ensure_batched(mask)
        model_input, shape = _flatten_antenna_grid(H_sparse_b, mask_b)
        residual = self.net(model_input)
        residual = _restore_antenna_grid(residual, shape)
        output = H_sparse_b + residual
        return output if batched else output.squeeze(0)


def _nearest_interpolate(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    observed_indices = torch.nonzero(observed, as_tuple=False).flatten()
    if observed_indices.numel() == 0:
        return values.clone()

    grid = torch.arange(values.shape[-1], device=values.device)
    distances = torch.abs(grid[:, None] - observed_indices[None, :])
    nearest_indices = observed_indices[torch.argmin(distances, dim=1)]
    return values[nearest_indices]


def _flatten_antenna_grid(
    H_sparse: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    batch_size, _, n_rx, n_tx, n_subcarriers = H_sparse.shape
    model_input = torch.cat([H_sparse, mask], dim=1)
    model_input = model_input.reshape(batch_size, 4, n_rx * n_tx, n_subcarriers)
    return model_input, (batch_size, n_rx, n_tx, n_subcarriers, H_sparse.shape[1])


def _restore_antenna_grid(
    tensor: torch.Tensor,
    shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    batch_size, n_rx, n_tx, n_subcarriers, output_channels = shape
    return tensor.reshape(batch_size, output_channels, n_rx, n_tx, n_subcarriers)


def _ensure_batched(tensor: torch.Tensor) -> tuple[bool, torch.Tensor]:
    if tensor.ndim == 5:
        return True, tensor
    if tensor.ndim == 4:
        return False, tensor.unsqueeze(0)
    raise ValueError(
        "CSI tensors must have shape (B, 2, n_rx, n_tx, n_subcarriers) "
        "or (2, n_rx, n_tx, n_subcarriers)."
    )


def _validate_csi_pair(H_sparse: torch.Tensor, mask: torch.Tensor) -> None:
    if not isinstance(H_sparse, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("H_sparse and mask must be torch.Tensor instances.")
    if H_sparse.shape != mask.shape:
        raise ValueError(f"H_sparse and mask must have the same shape, got {H_sparse.shape} and {mask.shape}.")
    if H_sparse.shape[-4] != 2:
        raise ValueError("CSI tensors must use two real-imaginary channels.")
