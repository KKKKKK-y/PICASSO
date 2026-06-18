# Final Project Summary

## Project

PICASSO: Physics-Informed Channel Synthesis from Sparse Pilot Observations

The repository implements a staged, lightweight, reproducible CSI reconstruction study for sparse pilot observations in OFDM/MIMO-OFDM systems.

## Stage 0 To Stage 4 Summary

- Stage 0/1A established the package skeleton, synthetic MIMO-OFDM generation, sparse pilot masks, physics losses, and smoke-level baselines.
- Stage 1B added classical and neural baselines: LS, LMMSE-like, OMP-like, CNN, and DnCNN.
- Stage 1C integrated PICASSO generator/discriminator skeletons and composite adversarial/physics losses as smoke tests.
- Stage 2A introduced noisy pilot observations and a compact formal grid across pilot ratios and SNR.
- Stage 2B/2C expanded diagnostics with random channel difficulty, condition-aware inputs, stronger baselines, ablations, and paper-style tables.
- Stage 3A shifted focus to supervised physics-guided reconstruction before any long full-GAN training.
- Stage 3A-L scaled model capacity and training budget in a controlled diagnostic.
- Stage 3B added incremental structural enhancements and stronger physics-consistency losses.
- Stage 4 introduced CDL-inspired realistic channel generalization with CDL-A/B/C, mobility/Doppler, pilot-pattern variation, and pilot contamination.

## Final Model Ranking

The final release default is **PICASSO-rec**.

- Primary model: PICASSO-rec
- Secondary ablation: PICASSO-rec-physics
- Strong neural baseline: Enhanced-DnCNN
- Classical baselines: LS, LMMSE-like, OMP-like
- Experimental only: PICASSO-full, PICASSO-cond-full, feature/output GAN variants

## CDL And Mobility Inclusion

Stage 4 adds a self-contained CDL-inspired simulator with clustered delay profiles, antenna steering, delay-spread variation, pilot contamination, pilot pattern variation, and velocity-dependent Doppler phase. The implementation is lightweight and reproducible, not a full 3GPP TR 38.901 standard simulator.

## GAN Conclusion

GAN training is not beneficial enough to be the primary release path. Stage 1C-3B keep adversarial components traceable, but Stage 4 disables GAN as a main experiment. The final recommendation is to keep GAN only as optional historical ablation code.

## Physics Conclusion

Physics losses are useful for consistency analysis and can regularize reconstruction, but the final measured gain is limited. PICASSO-rec-physics remains a secondary model and ablation, not the default.

## Final Recommendation

Use PICASSO-rec as the main method for paper writing and further reproducible experiments. Compare against LS, LMMSE-like, OMP-like, DnCNN, and Enhanced-DnCNN. Report PICASSO-rec-physics as a physics ablation. Keep GAN modules available for traceability, but do not present adversarial training as the central contribution unless future realistic datasets show a clear reversal.
