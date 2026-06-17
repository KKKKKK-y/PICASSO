# Method Design

## Objective

PICASSO is planned as a physics-informed generative framework for sparse-pilot CSI reconstruction. The first target is MIMO-OFDM:

`low-pilot LS CSI + pilot mask -> full CSI`

## Planned Architecture

The eventual model will contain:

- Generator: maps sparse CSI observations to full CSI.
- Discriminator: distinguishes reconstructed CSI from simulated full CSI.
- Physics-informed loss module: adds wireless-channel constraints to the learning objective.

Stage 0 does not implement the generator, discriminator, or training loop. It only defines simulator and loss interfaces that future Stage 1 code can call.

## Tensor Convention

All Stage 0 tensors use real-imaginary two-channel format:

`(n_rx, n_tx, n_subcarriers, 2)`

For batched training in later stages, the expected extension is:

`(batch_size, n_rx, n_tx, n_subcarriers, 2)`

The Stage 0 losses should accept either unbatched or batched tensors where practical.

## Physics-Informed Losses

### Pilot Consistency Loss

The reconstructed CSI must agree with observed pilot positions:

`L_pilot = MSE(mask * H_hat, mask * H_sparse)`

This is the most important physics/data-consistency term for the first paper.

### Frequency Smoothness Loss

Adjacent subcarriers should be locally correlated in ordinary multipath channels:

`L_smooth = mean(|H_hat[k+1] - H_hat[k]|^2)`

This term should be used carefully so it does not oversmooth frequency-selective fading.

### Delay Sparsity Loss

A multipath channel is sparse or compressible in the delay domain. The planned proxy is:

1. Convert real-imaginary CSI to complex form.
2. Apply IFFT along the subcarrier axis.
3. Penalize average delay-domain magnitude.

Stage 0 implements this as a differentiable PyTorch scalar loss.

## Ablation Plan

Minimum ablations for the first paper:

- w/o physics loss
- w/o adversarial loss
- w/o pilot consistency loss
- w/o sparsity constraint

## Non-Goals

Stage 0 does not:

- train a GAN
- generate saved datasets
- implement full baselines
- save model checkpoints
- run large-scale experiments
