# Stage 4 CDL Generalization Report

Grid type: reduced
Reduction: estimated units 5529600 exceeded threshold
Runtime seconds: 331.62

## CDL vs Synthetic Comparison
Stage 4 replaces the earlier simple synthetic multipath generator with a CDL-inspired clustered channel model. The new data includes CDL-A/B/C profiles, spatial steering, delay-spread variation, pilot contamination noise, pilot patterns, and velocity-dependent Doppler phase.

## Overall NMSE
- PICASSO-rec: 0.81130922
- PICASSO-rec-physics: 0.82301706
- Enhanced-DnCNN: 0.85062706
- LS: 0.85126823
- LMMSE-like: 1.73967190

## Mobility Impact
- Best static-channel method: PICASSO-rec
- Best high-mobility method: PICASSO-rec
Doppler robustness is recorded as reconstruction error normalized by temporal channel variation.

## Required Questions
1. PICASSO vs DnCNN under CDL: yes (PICASSO-rec=0.81130922, Enhanced-DnCNN=0.85062706).
2. Physics loss under mobility: no (PICASSO-rec-physics=0.82301706, PICASSO-rec=0.81130922).
3. GAN usefulness: not supported in Stage 4 main experiments; GAN remains disabled by policy after Stage 3B showed weak or negative gains.
4. Doppler sensitivity: compare velocity rows in stage4_summary.csv; higher velocity generally raises the Doppler-normalized error.
5. Condition-aware modeling: not promoted here; velocity is recorded as an input condition for Stage 5 but main Stage 4 models intentionally avoid increasing architecture.
6. Synthetic-stage conclusion: stability of supervised PICASSO remains the key question under CDL; GAN remains unnecessary unless later realistic data contradicts this result.

## Stage 5 Recommendation
Move toward paper writing only after a slightly larger CDL run confirms the same ordering. If PICASSO-rec remains competitive, Stage 5 should emphasize supervised physics-guided reconstruction under realistic CDL mobility rather than GAN synthesis.
