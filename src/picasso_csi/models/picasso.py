"""PICASSO generator and discriminator models."""

from __future__ import annotations

import torch
from torch import nn

from picasso_csi.models.conditioning import make_condition_channels


class PICASSOGenerator(nn.Module):
    """Residual generator for sparse-pilot CSI reconstruction."""

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 6,
        use_residual_blocks: bool = True,
        use_skip_connections: bool = True,
        use_condition: bool = False,
        refinement_blocks: int = 0,
        use_multiscale_fusion: bool = False,
        use_channel_attention: bool = False,
        use_film_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        if refinement_blocks < 0:
            raise ValueError("refinement_blocks must be non-negative.")

        self.use_condition = bool(use_condition)
        self.use_skip_connections = bool(use_skip_connections)
        self.use_film_conditioning = bool(use_film_conditioning)
        if self.use_condition and input_channels == 2:
            input_channels = 4
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if use_multiscale_fusion:
            layers.append(_MultiScaleFusion(base_channels))
        for _ in range(num_blocks):
            layers.append(_ResidualBlock(base_channels) if use_residual_blocks else _ConvBlock(base_channels))
        for _ in range(refinement_blocks):
            layers.append(_ResidualBlock(base_channels))
        if use_channel_attention:
            layers.append(_SEBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.film = nn.Linear(2, base_channels * 2) if self.use_film_conditioning else None

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
        if self.film is None:
            residual = self.net(model_input)
        else:
            residual = self._forward_with_film(model_input, snr_db, pilot_ratio)
        residual = residual.reshape(batch_size, channels, n_rx, n_tx, n_subcarriers)
        output = H_sparse_b + residual if self.use_skip_connections else residual
        return output if batched else output.squeeze(0)

    def _forward_with_film(
        self,
        model_input: torch.Tensor,
        snr_db: float | torch.Tensor | None,
        pilot_ratio: float | torch.Tensor | None,
    ) -> torch.Tensor:
        if snr_db is None or pilot_ratio is None:
            raise ValueError("snr_db and pilot_ratio are required when use_film_conditioning=True.")
        batch_size = model_input.shape[0]
        device = model_input.device
        snr = torch.as_tensor(snr_db, dtype=torch.float32, device=device).flatten()
        ratio = torch.as_tensor(pilot_ratio, dtype=torch.float32, device=device).flatten()
        if snr.numel() == 1:
            snr = snr.expand(batch_size)
        if ratio.numel() == 1:
            ratio = ratio.expand(batch_size)
        cond = torch.stack([torch.clamp(snr / 40.0, 0.0, 1.0), torch.clamp(ratio, 0.0, 1.0)], dim=1)
        gamma_beta = self.film(cond)
        first_conv = self.net[0](model_input)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        x = first_conv * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        x = self.net[1](x)
        for layer in self.net[2:]:
            x = layer(x)
        return x


class PICASSODiscriminator(nn.Module):
    """Optional legacy GAN discriminator for ablation experiments.

    Final-release experiments keep PICASSO-rec as the default supervised
    reconstruction model; adversarial training is retained only for traceable
    Stage 1C-3B diagnostics.
    """

    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        use_delay_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_delay_features = bool(use_delay_features)
        if self.use_delay_features:
            input_channels *= 2
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(base_channels * 4, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        batched, H_b = _ensure_batched(H)
        batch_size, channels, n_rx, n_tx, n_subcarriers = H_b.shape
        model_input = H_b.reshape(batch_size, channels, n_rx * n_tx, n_subcarriers)
        if self.use_delay_features:
            delay = torch.fft.ifft(torch.complex(H_b[:, 0], H_b[:, 1]), dim=-1)
            delay_input = torch.stack([delay.real, delay.imag], dim=1).reshape(
                batch_size, channels, n_rx * n_tx, n_subcarriers
            )
            model_input = torch.cat([model_input, delay_input], dim=1)
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


class _ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _MultiScaleFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.branch_3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.branch_1(x), self.branch_3(x)], dim=1)
        return self.project(fused) + x


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


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
