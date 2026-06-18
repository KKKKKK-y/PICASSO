"""Optional legacy adversarial losses for PICASSO smoke/ablation training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_bce_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy discriminator loss with logits."""

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
    return 0.5 * (real_loss + fake_loss)


def generator_bce_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy generator loss with logits."""

    targets = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, targets)
