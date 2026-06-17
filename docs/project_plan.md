# Project Plan

## Motivation

Sparse-pilot CSI reconstruction is a practical way to reduce pilot overhead in OFDM and MIMO-OFDM systems. Purely data-driven reconstruction methods can recover missing CSI, but they often require large training sets and may violate physical channel structure.

## Problem Definition

Given low-pilot LS channel estimates and a pilot mask, reconstruct the full CSI over the OFDM resource grid. The first-stage task uses pilot ratios of `1/2`, `1/4`, `1/8`, and `1/16`.

## Candidate Scenarios

- MIMO-OFDM sparse-pilot CSI reconstruction
- Massive MIMO CSI recovery
- RIS-assisted channel estimation
- XL-MIMO near-field channel estimation

## Final Recommended Scenario

The first paper should focus on MIMO-OFDM sparse-pilot CSI reconstruction. This scenario has a clear input-output definition, mature baselines, simple simulation requirements, and a compact story suitable for IEEE Communications Letters.

## Initial Method Design

PICASSO is planned as a physics-informed GAN. The generator reconstructs full CSI from sparse pilot observations. The discriminator encourages realistic channel synthesis. Physics-informed losses constrain pilot consistency, OFDM time-frequency smoothness, and multipath sparsity.

## Planned Baselines

Required baselines:

- LS
- LMMSE
- OMP
- CNN
- DnCNN
- GAN without physics
- PINN without GAN
- Proposed PICASSO

Optional baselines:

- CsiNet
- CRNet
- Transformer

## Planned Metrics

- NMSE
- BER after equalization
- Reconstruction quality across pilot ratios
- SNR generalization
- Parameter count
- Inference time
- Ablation results for physics, adversarial, and pilot-consistency losses

## Next Steps

- Freeze the simulator specification.
- Define pilot masks and channel model settings.
- Implement lightweight data generation scripts.
- Implement baseline estimators.
- Implement the proposed model after the experimental route is confirmed.
