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

Literature survey and scenario selection stage. No training code or dataset is included yet.

## Artifact Policy

- Do not commit raw datasets.
- Do not commit model checkpoints.
- Do not commit generated outputs.
- Keep only configuration files, source code, documentation, and lightweight survey files.
