# Stage 5A Paper-Level Ablation Report

## 1. Ablation Study Objective

This stage converts the existing experiments into a traceable ablation argument. It does not redesign the model or overwrite any Stage 0-4 result.

## 2. Existing Result Reuse Strategy

Loaded row-level sources: stage2a, stage2bc, stage3a_larger, stage3b, stage4. Historical rows retain source provenance and are never relabeled as newly run results.

## 3. Newly Run Experiments

New reduced-grid rows: 30. They isolate architecture and physics-loss components at pilot ratio 0.125, 10 dB, two seeds, and two epochs. They support mechanism analysis only.

## 4. Model Ablation Results

- PICASSO-rec: mean NMSE 0.89559736
- PICASSO-rec-physics: mean NMSE 0.89635902
- PICASSO-cond-full: mean NMSE 0.89640709
- PICASSO-full: mean NMSE 0.89647962
- Enhanced-DnCNN: mean NMSE 0.89761454
- DnCNN: mean NMSE 0.89841003
- LS: mean NMSE 0.91119584

## 5. Architecture Ablation Results

- PICASSO-rec-attn: mean NMSE 0.88508414
- PICASSO-rec-SE: mean NMSE 0.88595106
- PICASSO-rec-multiscale: mean NMSE 0.88760105
- PICASSO-rec-enhanced: mean NMSE 0.88778854
- PICASSO-rec-base: mean NMSE 0.88872317
- PICASSO-rec-FiLM: mean NMSE 0.89023005
- PICASSO-rec-refinement: mean NMSE 0.89123694

## 6. Loss Component Ablation Results

- rec + pilot + delay: mean NMSE 0.88502013
- rec only: mean NMSE 0.88532707
- rec + energy preservation: mean NMSE 0.88541611
- rec + frequency consistency: mean NMSE 0.88729804
- rec + pilot consistency: mean NMSE 0.88864707
- rec + delay-domain sparsity: mean NMSE 0.89151723
- rec + pilot + frequency: mean NMSE 0.89199587
- rec + full physics: mean NMSE 0.89227996

## 7. GAN Ablation Results

- no GAN (Stage 3B): mean NMSE 0.75949831
- feature-level GAN (Stage 3B): mean NMSE 0.76158459
- conditioned feature-level GAN (Stage 3B): mean NMSE 0.76614973
- output-level GAN (Stage 2BC): mean NMSE 0.85438569
- no GAN (Stage 2BC): mean NMSE 0.85754549
- light adversarial loss (Stage 2BC): mean NMSE 0.85757674

## 8. Channel Complexity Ablation Results

- noisy synthetic OFDM: mean NMSE 0.77596677
- noisy random-path synthetic OFDM: mean NMSE 0.86067338
- CDL-B: mean NMSE 0.98208765
- CDL-C: mean NMSE 1.01514134
- CDL-A: mean NMSE 1.04830709

## 9. Robustness Ablation Results

- PICASSO-rec: mean NMSE 0.81130922
- PICASSO-rec-physics: mean NMSE 0.82301706
- Enhanced-DnCNN: mean NMSE 0.85062706
- LS: mean NMSE 0.85126823

## 10. Final Model Choice

PICASSO-rec remains the final model. It is the most stable supervised choice across the controlled synthetic and CDL-inspired evidence; reduced component diagnostics do not replace that broader evidence.

## 11. Why PICASSO-rec Is Final

Its gain is attributable primarily to reconstruction architecture rather than adversarial training or a larger loss stack. It also avoids the instability and additional optimization state of GAN variants.

## 12. Why GAN Is Optional/Deprecated

Within-stage paired controls show that neither output/light adversarial nor feature-level adversarial variants deliver a consistent NMSE gain. Stage 2BC and Stage 3B absolute values are not compared directly. Distributional effects were not measured with a validated perceptual metric, so no stronger GAN claim is warranted.

## 13. Why Physics Loss Is Secondary

Component isolation identifies `rec + full physics` as the weakest reduced-grid loss setting, while bundled physics is also worse than reconstruction-only in Stage 3B and Stage 4. Physics remains useful for consistency analysis, but over-regularization and simulator mismatch limit its main-metric gain.

## 14. Strongest PICASSO Condition

The clearest broad advantage appears in CDL-inspired reconstruction, including sparse-pilot and mobility conditions, where Stage 4 reports PICASSO-rec ahead of Enhanced-DnCNN overall. Exact gains should be quoted only from matched Stage 4 rows.

## 15. Suggested Final Paper Tables

- **Table I - Main Performance Comparison:** Method, Synthetic NMSE, CDL NMSE, Low Pilot NMSE, High Mobility NMSE, Avg NMSE.
- **Table II - Ablation Study:** Variant, Removed/Added Component, NMSE, Relative Change, Interpretation.
- **Table III - Robustness Analysis:** Condition, LS, DnCNN, Enhanced-DnCNN, PICASSO-rec, Gain over Best Baseline.

## 16. Suggested Final Paper Figures

- **Figure 1 - PICASSO Framework:** sparse pilot CSI, reconstruction network, optional physics loss, full CSI output.
- **Figure 2 - NMSE vs Pilot Ratio:** LS, DnCNN, Enhanced-DnCNN, PICASSO-rec.
- **Figure 3 - NMSE vs Velocity:** Enhanced-DnCNN, PICASSO-rec, PICASSO-rec-physics.
- **Figure 4 - Ablation Bar Chart:** reconstruction, reconstruction+physics, reconstruction+GAN, enhanced reconstruction.

## 17. Limitations

The CDL simulator is CDL-inspired rather than a complete TR 38.901 implementation. Several tables combine provenance-aware historical stages with different budgets, so cross-stage values are descriptive. The component runs are deliberately small, and no claim of statistical significance is made. Real measured CSI remains future validation.

## Final Evidence Statement

The strongest isolated architecture variant is `PICASSO-rec-attn`, but the complete evidence still supports PICASSO-rec as final, physics as secondary, and GAN as deprecated.
