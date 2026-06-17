# Data Policy

This directory is reserved for dataset notes and lightweight metadata.

Do not commit raw datasets, processed arrays, generated CSI tensors, model checkpoints, or training outputs.

Recommended first-stage data route:

- Start with a self-built MIMO-OFDM simulator.
- Generate sparse pilot observations at pilot ratios `1/2`, `1/4`, `1/8`, and `1/16`.
- Use the complete simulated CSI grid as the label.
- Keep generated arrays outside Git, under ignored local folders such as `data/raw/`, `data/processed/`, or `data/generated/`.
