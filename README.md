# PICASSO-CSI

Physics-Informed Channel Synthesis from Sparse Observations for CSI Reconstruction

## Overview

PICASSO-CSI is a research repository for sparse-pilot CSI reconstruction in OFDM / MIMO-OFDM systems. The project studies whether a physics-informed generative model can reconstruct full channel state information from a small number of pilot observations.

## Research Objective

The first paper target is a lightweight and reproducible IEEE Communications Letters-style study on sparse-pilot CSI reconstruction. The intended task is:

- Input: low-pilot LS channel estimates, pilot masks, and optional SNR/noise information.
- Output: full CSI over the OFDM resource grid.
- Goal: reduce pilot overhead while preserving reconstruction accuracy and physical consistency.

## Target Paper

Working title:

PICASSO: Physics-Informed Channel Synthesis from Sparse Pilot Observations

Target journal:

IEEE Communications Letters

## Proposed Method

The planned method combines:

- Sparse-pilot CSI reconstruction
- GAN-based channel synthesis
- Physics-informed constraints
- OFDM / MIMO-OFDM channel model
- Pilot consistency loss
- Time-frequency smoothness loss
- Multipath sparsity prior

## Repository Structure

- `configs/`: lightweight experiment configuration files.
- `data/`: dataset notes and placeholders only; no raw data should be committed.
- `docs/`: project planning and literature summaries.
- `literature/`: survey and scenario-selection spreadsheets.
- `notebooks/`: exploratory notebooks, if needed later.
- `scripts/`: command-line utilities, if needed later.
- `src/picasso_csi/`: Python package skeleton.
- `tests/`: future tests.
- `outputs/`: local generated outputs; ignored except for `.gitkeep`.

## Current Status

Stage 0 and Stage 1A are complete. The repository now includes lightweight
MIMO-OFDM simulation interfaces, an in-memory synthetic CSI dataset, basic
physics losses, CNN/DnCNN smoke baselines, and smoke-level metrics.

Stage 1B adds a baseline evaluation suite for LS, LMMSE-like, OMP-like, CNN, and
DnCNN baselines. These runs are code-path validation only; they are not formal
paper experiments.

Stage 1C adds PICASSO generator/discriminator skeletons, adversarial and
physics-informed generator losses, smoke training, and multi-pilot-ratio smoke
evaluation. These are integration checks, not final GAN results.

Stage 2A adds noise-aware sparse pilot observations and a small formal
experiment protocol across pilot ratios, SNR values, and LS/DnCNN/PICASSO
methods. It saves only a compact metrics CSV.

Stage 2B-2C adds a comprehensive diagnostic experiment pack with random channel
difficulty controls, condition-aware inputs, stronger DnCNN/PICASSO variants,
multi-seed reduced-grid evaluation, loss ablations, paper-style CSV tables, and
Markdown analysis. These runs are still small-scale diagnostics, not final paper
results.

Run the Stage 1B smoke evaluation from the repository root:

```powershell
conda run -n picasso python src/picasso_csi/training/evaluate_baselines.py --config configs/stage1b_baselines.yaml
```

Run the Stage 1C PICASSO smoke loop:

```powershell
conda run -n picasso python src/picasso_csi/training/smoke_train_picasso.py --config configs/stage1c_picasso_smoke.yaml
```

Run the Stage 2A small formal protocol:

```powershell
conda run -n picasso python src/picasso_csi/training/run_stage2a_small_formal.py --config configs/stage2a_small_formal.yaml
```

Run the Stage 2B-2C comprehensive diagnostic protocol:

```powershell
conda run -n picasso python src/picasso_csi/training/run_stage2bc_comprehensive.py --config configs/stage2bc_comprehensive.yaml
```

Stage 2B-2C features:

- Condition-aware training with normalized SNR and pilot-ratio channels.
- Random path count, random delay spread, normalized synthetic channels, and pilot-only AWGN.
- LS, DnCNN, Cond-DnCNN, Enhanced-DnCNN, and PICASSO loss-mode variants.
- Reduced-grid fallback when the full grid is estimated to exceed the runtime budget.
- Lightweight CSV outputs under `outputs/results/*.csv` and Markdown reports under `docs/`.

## Artifact Policy

- Do not commit raw datasets.
- Do not commit model checkpoints.
- Do not commit generated outputs.
- Stage 2A may commit only compact CSV metrics under `outputs/results/*.csv`.
- Stage 2B-2C may commit only compact CSV metrics under `outputs/results/*.csv`.
- Keep only configuration files, source code, documentation, and lightweight survey files.
