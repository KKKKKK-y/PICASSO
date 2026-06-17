# Stage 2A Noise-Aware Small Formal Experiment

## Goal

Stage 2A defines the first small formal experiment protocol for PICASSO. It adds
noise-aware sparse pilot observations, fixed training budgets, LS/DnCNN/PICASSO
comparisons, and a compact CSV metrics table.

## Why Add SNR And Noise

Earlier stages validated clean sparse-pilot reconstruction code paths. A
communication-oriented CSI reconstruction experiment also needs noisy pilot
observations. Stage 2A keeps the full CSI label clean while adding Gaussian
noise only to observed pilot entries.

## Data Generation Protocol

The synthetic MIMO-OFDM channel generator remains in memory. For each sample:

1. Generate a clean full CSI tensor.
2. Create a sparse pilot mask.
3. Add Gaussian noise only at pilot positions.
4. Return clean `H_full`, noisy `H_sparse`, `mask`, `snr_db`, and `pilot_ratio`.

No generated training data is written to disk.

## Pilot Ratios

The protocol evaluates:

- 0.5
- 0.25
- 0.125
- 0.0625

## SNR Settings

The protocol evaluates:

- 10 dB
- 20 dB
- 30 dB

Noise power follows:

`noise_power = signal_power / (10 ** (snr_db / 10))`

## Methods

- LS: direct noisy sparse observation baseline.
- DnCNN: residual CNN trained under the fixed budget.
- PICASSO: generator/discriminator skeleton trained under the fixed budget.

## Metrics

The result CSV records NMSE, MSE, MAE, pilot consistency error, and
delay-domain sparsity score for each method, pilot ratio, and SNR.

## Artifact Policy

Stage 2A may save only:

`outputs/results/stage2a_small_formal_results.csv`

It must not save checkpoints, NumPy arrays, generated datasets, model outputs,
or compressed artifacts.

## Run Command

```powershell
conda run -n picasso python src/picasso_csi/training/run_stage2a_small_formal.py --config configs/stage2a_small_formal.yaml
```

## Limitations

This is still a small synthetic protocol. The CSV values are suitable for
pipeline validation and early comparison, not final paper claims. The PICASSO
architecture is still a skeleton from Stage 1C.

## Stage 2B Plan

- Add controlled repetitions across multiple seeds.
- Add SNR-aware model inputs or conditioning.
- Add validation-based model selection without saving checkpoints by default.
- Compare stronger DnCNN/PICASSO variants under the same budget.
- Prepare figures and tables from the compact CSV metrics.
