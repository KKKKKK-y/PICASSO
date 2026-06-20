# Stage 5A Paper-Level Ablation Design

## Principle

The study uses the smallest experiment set that closes a specific evidence gap. Every output row records its source stage and whether it was reused or newly run.

## Six-Layer Design

### Layer 1: Model Variants

Compare LS, DnCNN, Enhanced-DnCNN, PICASSO-rec, PICASSO-rec-enhanced, PICASSO-rec-physics, PICASSO-full, and PICASSO-cond-full. Report overall and difficult-condition NMSE, MSE, MAE, runtime, and parameter counts where the originating stage recorded them.

### Layer 2: Architecture

Compare base, refinement-only, multiscale-only, SE-only, combined enhanced, attention-enhanced, and FiLM-conditioned variants. Stage 3B rows are reused; missing isolated components are run at pilot ratio 0.125 and 10 dB with seeds 42 and 123.

### Layer 3: Loss Components

Use the same enhanced generator and compare reconstruction only; pilot, frequency, delay sparsity, or energy individually; pilot+frequency; pilot+delay; and full physics. Only loss weights change. This is a reduced diagnostic for mechanism attribution, not a new main benchmark.

### Layer 4: GAN

Reuse no-GAN, output/light adversarial, feature-level, and conditioned feature-level rows. No additional GAN training is justified because all completed stages show weak or negative NMSE changes.

### Layer 5: Channel Complexity

Retain noisy/random-path synthetic rows and CDL-A/B/C plus mobility rows. Results are grouped by their source stage to avoid implying identical data distributions.

### Layer 6: Robustness

Reuse pilot ratios 0.5/0.25/0.125/0.0625, SNR 0/10/30 dB, velocities 0/60/120 km/h, and comb/block/irregular patterns. Report gain over the best available baseline only within matched conditions.

## Decision Rules

- Lower NMSE is better; relative change is `(variant - reference) / reference`.
- A small mean gain without consistent seed-wise direction is described as inconclusive.
- PICASSO-rec remains final unless a controlled variant improves NMSE without greater instability or an incompatible deployment cost.
- GAN remains deprecated unless it improves matched-condition NMSE consistently.
- Physics remains secondary unless component isolation and realistic-channel evidence both support it.

## Runtime And Artifacts

The supplemental grid uses 256 training and 128 test samples, two epochs, two seeds, and one representative condition. No checkpoints, datasets, arrays, or predictions are written; only compact CSV and Markdown files are retained.
