"""Shape tests for PICASSO Stage 0 interfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from picasso_csi.losses import (  # noqa: E402
    delay_sparsity_loss,
    frequency_smoothness_loss,
    pilot_consistency_loss,
)
from picasso_csi.simulation import (  # noqa: E402
    apply_pilot_mask,
    create_pilot_mask,
    generate_mimo_ofdm_channel,
)


def test_stage0_shapes_and_losses() -> None:
    n_tx = 4
    n_rx = 4
    n_subcarriers = 64
    n_paths = 6
    pilot_ratio = 0.25

    H_full = generate_mimo_ofdm_channel(
        n_tx=n_tx,
        n_rx=n_rx,
        n_subcarriers=n_subcarriers,
        n_paths=n_paths,
        seed=42,
    )
    mask = create_pilot_mask(
        n_rx=n_rx,
        n_tx=n_tx,
        n_subcarriers=n_subcarriers,
        pilot_ratio=pilot_ratio,
    )
    H_sparse = apply_pilot_mask(H_full, mask)

    expected_shape = (n_rx, n_tx, n_subcarriers, 2)
    assert H_full.shape == expected_shape
    assert mask.shape == expected_shape
    assert H_sparse.shape == expected_shape

    assert H_full.dtype == np.float32
    assert mask.dtype == np.float32
    assert H_sparse.dtype == np.float32

    pilot_positions = mask == 1.0
    non_pilot_positions = mask == 0.0
    assert np.allclose(H_sparse[pilot_positions], H_full[pilot_positions])
    assert np.allclose(H_sparse[non_pilot_positions], 0.0)

    H_hat = torch.from_numpy(H_full.copy())
    H_sparse_t = torch.from_numpy(H_sparse)
    mask_t = torch.from_numpy(mask)

    losses = [
        pilot_consistency_loss(H_hat, H_sparse_t, mask_t),
        frequency_smoothness_loss(H_hat),
        delay_sparsity_loss(H_hat),
    ]

    for loss in losses:
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


if __name__ == "__main__":
    test_stage0_shapes_and_losses()
    print("Stage 0 shape test passed.")
