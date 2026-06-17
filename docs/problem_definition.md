# Problem Definition

## Main Direction

PICASSO-CSI focuses on wireless channel synthesis and sparse-pilot CSI reconstruction.

The first paper should stay on:

MIMO-OFDM sparse-pilot CSI reconstruction

## Task Mapping

Sparse pilot observations -> full CSI reconstruction

## Input

The planned input is a low-pilot LS channel estimate together with its pilot mask. Optional metadata may include SNR or noise variance.

Pilot ratios for the first reproducible route:

- `1/2`
- `1/4`
- `1/8`
- `1/16`

## Output

The output is the full complex CSI over the MIMO-OFDM resource grid.

## Physics-Informed Constraints

The physics constraints are wireless-channel constraints for OFDM / MIMO-OFDM propagation:

- Pilot consistency
- OFDM time-frequency smoothness
- Multipath delay-domain sparsity
- Optional delay-domain consistency
- MIMO spatial correlation

## Proposed Method Scope

The planned method is PINN + GAN:

- Generator: reconstructs full CSI from sparse pilot observations.
- Discriminator: distinguishes real CSI from generated CSI.
- Physics-informed losses: enforce pilot consistency, smoothness, and sparsity.

No model training code, dataset, checkpoint, or generated output is included at this stage.
