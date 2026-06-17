# Stage 2BC Results Analysis

## Experiment Scope
- Scope: reduced
- Reason: full grid estimated units 26542080 exceeded the 5-hour diagnostic budget
- Runtime seconds: 28.85
- Seeds: [42, 123]
- Pilot ratios: [0.25, 0.125, 0.0625]
- SNR values: [10, 20]
- Train/test samples: 512 / 256
- Epochs main/diagnostic: 3 / 2

## Dataset Difficulty
- The dataset uses random path counts, random delay spread, complex Gaussian gains, normalized clean channels, and pilot-only AWGN.
- Labels remain clean H_full tensors, while H_sparse stores noisy pilot observations and zeros elsewhere.

## Overall Results
- Cond-DnCNN: NMSE 0.86170453 +/- 0.07807674 over 12 rows
- DnCNN: NMSE 0.86174773 +/- 0.07760515 over 12 rows
- Enhanced-DnCNN: NMSE 0.86190357 +/- 0.07760924 over 12 rows
- LS: NMSE 0.86214598 +/- 0.07728318 over 12 rows
- PICASSO-cond-full: NMSE 0.85484274 +/- 0.08237586 over 12 rows
- PICASSO-full: NMSE 0.85438569 +/- 0.08327666 over 12 rows
- PICASSO-rec: NMSE 0.85754549 +/- 0.08006160 over 12 rows
- PICASSO-rec-adv: NMSE 0.85757674 +/- 0.08144135 over 12 rows
- PICASSO-rec-physics: NMSE 0.86002412 +/- 0.07910645 over 12 rows

## Low Pilot Ratio Results
- Best method for pilot_ratio <= 0.125: PICASSO-full

## Low SNR Results
- Best method for SNR <= 10 dB: PICASSO-cond-full

## Condition-Aware Effect
- DnCNN vs Cond-DnCNN winner: Cond-DnCNN
- PICASSO-full vs PICASSO-cond-full winner: PICASSO-full

## Physics Loss Effect
- PICASSO-rec vs PICASSO-rec-physics winner: PICASSO-rec

## Adversarial Loss Effect
- PICASSO-rec vs PICASSO-rec-adv winner: PICASSO-rec
- PICASSO-rec-physics vs PICASSO-full winner: PICASSO-full

## Does PICASSO Beat Baselines?
- PICASSO beats LS: True
- PICASSO beats DnCNN: True
- PICASSO beats Enhanced-DnCNN: True
- These answers are based on the best overall PICASSO-family mean NMSE versus the listed baseline mean NMSE.

## Honest Diagnosis
- If the GAN variants do not lead the table, the current evidence favors supervised reconstruction first.
- Likely bottlenecks are task simplicity, zero-filled LS strength at observed pilots, short diagnostic training, and adversarial instability under small budgets.
- Physics losses can help only if their weights match the channel generator; otherwise pilot and sparsity penalties may over-constrain reconstruction.

## Stage 3 Recommendation
- Do not enter long formal GAN training until a supervised or physics-guided variant clearly beats Enhanced-DnCNN on the synthetic diagnostic.
- Prioritize supervised physics-guided reconstruction and stronger realistic channel data.
- Add 3GPP CDL or QuaDRiGa channels before using this as an IEEE Communications Letters paper result.
