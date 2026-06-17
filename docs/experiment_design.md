# Experiment Design

## First Experiment Route

The first reproducible experiment should use a self-built MIMO-OFDM simulator. This keeps the first paper independent from private datasets and makes the sparse-pilot task easy to inspect.

## Default System Parameters

- `n_tx = 4`
- `n_rx = 4`
- `n_subcarriers = 64`
- `n_paths = 6`
- pilot ratios: `[0.5, 0.25, 0.125, 0.0625]`

## Data Construction

For each synthetic channel:

1. Generate full CSI `H_full`.
2. Create a pilot mask for a selected pilot ratio.
3. Apply the mask to obtain `H_sparse`.
4. Use `H_sparse` and the mask as model inputs.
5. Use `H_full` as the label.

Stage 0 only tests these shapes in memory. It does not save arrays to disk.

## Recommended Baselines

Minimum baselines for Stage 1:

- LS
- LMMSE
- OMP
- CNN
- DnCNN
- GAN without physics
- PINN without GAN
- Proposed PICASSO

Optional heavier baselines:

- Diffusion
- Transformer
- CsiNet / CRNet for a CSI-feedback variant

## Metrics

Planned metrics:

- NMSE
- BER after simple equalization
- performance vs. pilot ratio
- SNR generalization
- inference time
- parameter count
- ablation loss contributions

## Stage 0 Validation

The Stage 0 test checks:

- `H_full` shape
- pilot mask shape
- `H_sparse` shape
- pilot-position consistency
- scalar output from pilot consistency loss
- scalar output from frequency smoothness loss
- scalar output from delay sparsity loss

## Handoff to Windows RTX 5070 Ti

The Mac stage should stop after interface and shape tests. The Windows RTX 5070 Ti stage can then implement dataset generation, baselines, PICASSO models, training, and visualization.
