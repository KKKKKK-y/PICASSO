# Stage 3A-L Larger PICASSO Training

## Why Larger Models And Longer Training
Stage 2BC and Stage 3A showed promising PICASSO trends under short diagnostics. Stage 3A-L increases model capacity and trains longer to test whether those gains survive a controlled larger-budget run.

## Parameter Counts
- DnCNN: 29506
- Enhanced-DnCNN: 225090
- PICASSO Generator: 445506
- PICASSO Conditional Generator: 446658
- PICASSO Discriminator: 1108161

## Training Budget
- Grid type: reduced
- Reason: full grid estimated units 247726080 exceeded the 5-hour diagnostic budget
- Runtime seconds: 167.19
- Train/val/test samples: 1024 / 256 / 512
- Epochs: 10

## Data And Loss Setup
- Random path count, random delay spread, normalized synthetic channels, and pilot-only AWGN are enabled.
- Loss weights: adv=0.001, pilot=1.0, smooth=0.05, sparse=0.05.
- GAN warmup: 2 epochs with rec_physics before light adversarial training.

## Overall Results
- DnCNN: NMSE 0.89841003 +/- 0.03778343 over 8 rows
- Enhanced-DnCNN: NMSE 0.89761454 +/- 0.03795856 over 8 rows
- LS: NMSE 0.91119584 +/- 0.03203482 over 8 rows
- PICASSO-cond-full: NMSE 0.89640709 +/- 0.03810032 over 8 rows
- PICASSO-full: NMSE 0.89647962 +/- 0.03810953 over 8 rows
- PICASSO-rec: NMSE 0.89559736 +/- 0.03837500 over 8 rows
- PICASSO-rec-physics: NMSE 0.89635902 +/- 0.03810132 over 8 rows

## Low Pilot Ratio Results
Best method for pilot_ratio <= 0.125: PICASSO-rec.

## Low SNR Results
Best method for SNR <= 10 dB: PICASSO-rec.

## PICASSO-Full vs Enhanced-DnCNN
PICASSO-full beats Enhanced-DnCNN: True.

## Condition-Aware Low SNR
PICASSO-cond-full is better than PICASSO-full at low SNR: True.

## Stability And Adversarial Loss
PICASSO-rec-physics stable over Enhanced-DnCNN: True.
Adversarial improvement over PICASSO-rec-physics: False.

## Stage 3B Recommendation
Scale supervised/physics-guided PICASSO first; keep adversarial loss as an ablation until it proves useful.

## 3GPP CDL / QuaDRiGa
A larger synthetic diagnostic is still not enough for final paper claims. Stage 3B should introduce 3GPP CDL or QuaDRiGa before scaling formal experiments.
