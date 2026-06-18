# Stage 3A Supervised Physics-Guided Reconstruction

## Stage 3A Goal
Stage 3A tests supervised physics-guided CSI reconstruction before committing to long full-GAN training.

## Why Full GAN Is Deferred
Stage 2BC showed that adversarial variants can help in a reduced diagnostic, but the evidence is not yet strong enough for long GAN training. The key gate is whether supervised PICASSO-rec-physics can beat Enhanced-DnCNN.

## Why Supervised Physics First
A supervised reconstruction model is easier to train, easier to diagnose, and directly tests whether pilot consistency, frequency smoothness, and delay sparsity add value beyond a stronger CNN baseline.

## Experiment Setup
- Grid type: reduced
- Reason: full grid estimated units 70778880 exceeded the diagnostic budget
- Runtime seconds: 204.99
- Seeds: [42, 123, 2026]
- Pilot ratios: [0.5, 0.25, 0.125, 0.0625]
- SNR values: [10, 20, 30]
- Train/val/test samples: 1024 / 256 / 512
- Epochs and early-stop patience: 5 / 3

## Dataset Difficulty
The diagnostic uses random path counts, random delay spread, complex Gaussian path gains, normalized clean channels, and pilot-only AWGN.

## Model Comparison
Methods: LS, DnCNN, Enhanced-DnCNN, PICASSO-rec, PICASSO-rec-physics, PICASSO-full-light-adv.

## Loss Setup
- lambda_adv: 0.001
- lambda_pilot: 1.0
- lambda_smooth: 0.05
- lambda_sparse: 0.05

## Results Overview
- DnCNN: NMSE 0.75824519 +/- 0.17848203 over 36 rows
- Enhanced-DnCNN: NMSE 0.75029091 +/- 0.18409710 over 36 rows
- LS: NMSE 0.77419812 +/- 0.16398590 over 36 rows
- PICASSO-full-light-adv: NMSE 0.74146493 +/- 0.18496600 over 36 rows
- PICASSO-rec: NMSE 0.73889123 +/- 0.18674354 over 36 rows
- PICASSO-rec-physics: NMSE 0.74097347 +/- 0.18541561 over 36 rows

## Low Pilot Ratio Analysis
Best method for pilot_ratio <= 0.125: PICASSO-rec.

## Low SNR Analysis
Best method for SNR <= 10 dB: PICASSO-rec.

## PICASSO-rec-physics vs Enhanced-DnCNN
PICASSO-rec-physics beats Enhanced-DnCNN: True.

## PICASSO-full-light-adv vs PICASSO-rec-physics
PICASSO-full-light-adv beats PICASSO-rec-physics: False.

## Stage 3B Recommendation
Proceed to Stage 3B by scaling supervised physics-guided reconstruction first; keep GAN as a secondary ablation.

## 3GPP CDL / QuaDRiGA Recommendation
Introduce 3GPP CDL or QuaDRiGa before claiming paper-level realism. The synthetic generator is useful for method gating but not sufficient as final channel evidence.
