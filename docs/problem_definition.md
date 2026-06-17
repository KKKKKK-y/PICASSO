# Problem Definition

## Main Direction

PICASSO-CSI focuses on wireless channel synthesis and sparse-pilot CSI reconstruction. The first paper should stay on a compact MIMO-OFDM setting:

Sparse pilot observations -> full CSI reconstruction

## Stage 0 Scope

Stage 0 defines the task, tensor shapes, minimal simulator interfaces, and loss interfaces. It does not include model training, dataset generation, checkpoints, or generated experiment outputs.

## System Model

The minimum system is a narrow-scope MIMO-OFDM link with:

- `n_tx` transmit antennas
- `n_rx` receive antennas
- `n_subcarriers` OFDM subcarriers
- `n_paths` multipath components

The synthetic channel is represented as a complex frequency-domain CSI tensor. In code, complex values are stored with a final real-imaginary channel:

`H.shape = (n_rx, n_tx, n_subcarriers, 2)`

where:

- `H[..., 0]` is the real part
- `H[..., 1]` is the imaginary part

## Input

The planned model input is a sparse LS-style channel observation:

- `H_sparse`: sparse pilot observation with the same shape as `H`
- `mask`: pilot mask broadcast-compatible with `H`
- optional future metadata: SNR or noise variance

The first reproducible route uses pilot ratios:

- `1/2`
- `1/4`
- `1/8`
- `1/16`

## Output

The output is the reconstructed full CSI:

`H_hat.shape = (n_rx, n_tx, n_subcarriers, 2)`

The reconstruction target is the complete synthetic CSI `H_full`.

## Physics-Informed Constraints

The physics constraints are wireless-channel constraints for OFDM / MIMO-OFDM propagation:

- Pilot consistency: reconstructed CSI should match observed pilot positions.
- Frequency smoothness: adjacent subcarriers should not vary unrealistically.
- Multipath delay-domain sparsity: the delay-domain channel should be sparse or compressible.
- Optional delay-domain consistency: future versions may directly regularize delay-domain reconstructions.
- MIMO spatial correlation: future versions may add antenna-domain structural priors.

## Proposed Method Scope

The planned method is PINN + GAN:

- Generator: reconstructs full CSI from sparse pilot observations.
- Discriminator: distinguishes real CSI from generated CSI.
- Physics-informed losses: enforce pilot consistency, smoothness, and sparsity.

Stage 0 only exposes minimal interfaces for these losses. It intentionally avoids a full GAN training loop.
