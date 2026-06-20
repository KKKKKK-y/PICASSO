# Stage 5B Controlled Ablation Report

## 1. Purpose

Stage 5A consolidated historical evidence and included small gap runs. Stage 5B is a new strict controlled ablation: every reported row uses the same data protocol, channel condition, budget, optimizer, batch size, and evaluator, with one targeted module changed at a time.

## 2. Controlled Setting

- Channel profiles: CDL-A, CDL-B, CDL-C
- Pilot ratio: 0.125
- SNR: 10 dB
- Velocity: 60 km/h
- Pilot pattern: comb
- Seeds: [42, 123, 2026]
- Epochs: 30 for every trainable variant
- Optimizer / learning rate: Adam / 0.0001
- Batch size: 64
- Runtime: 858.64 seconds

## 3. Baseline Results

- PICASSO-rec-base: NMSE 0.84511966 +/- 0.01894520; delta +0.00000000; neutral
- Enhanced-DnCNN: NMSE 0.84845396 +/- 0.01503404; delta +0.00333429; negative
- LS: NMSE 0.88709148 +/- 0.00047117; delta +0.04197182; negative

The comparison is fully matched across CDL profile and seed. LS has no trainable budget by definition; both neural methods use the same training dataset and evaluator.

## 4. Architecture Ablation

- PICASSO-rec + refinement + multi-scale + SE: NMSE 0.84263315 +/- 0.02121661; delta -0.00248651; positive
- PICASSO-rec + refinement blocks only: NMSE 0.84313596 +/- 0.02121147; delta -0.00198371; positive
- PICASSO-rec + refinement + multi-scale: NMSE 0.84370249 +/- 0.02091823; delta -0.00141717; positive
- PICASSO-rec-base: NMSE 0.84511966 +/- 0.01894520; delta +0.00000000; neutral
- PICASSO-rec-enhanced full architecture: NMSE 0.84516710 +/- 0.01818932; delta +0.00004744; neutral
- PICASSO-rec + SE attention only: NMSE 0.84539868 +/- 0.01802481; delta +0.00027902; neutral
- PICASSO-rec + multi-scale fusion only: NMSE 0.84555482 +/- 0.01831536; delta +0.00043516; neutral
- PICASSO-rec + FiLM condition only: NMSE 0.84704654 +/- 0.01641915; delta +0.00192688; negative

- Refinement: positive (delta -0.00198371)
- Multi-scale: neutral (delta +0.00043516)
- SE attention: neutral (delta +0.00027902)
- FiLM: negative (delta +0.00192688)
- Best architecture: PICASSO-rec + refinement + multi-scale + SE
- Full enhanced architecture: neutral (delta +0.00004744).

## 5. Loss Ablation

- PICASSO-rec-full physics: NMSE 0.84492910 +/- 0.01721885; delta -0.00019056; neutral
- PICASSO-rec + delay-domain sparsity: NMSE 0.84498604 +/- 0.01896750; delta -0.00013362; neutral
- PICASSO-rec only: NMSE 0.84511966 +/- 0.01894520; delta +0.00000000; neutral
- PICASSO-rec + frequency consistency: NMSE 0.84520747 +/- 0.01877968; delta +0.00008781; neutral
- PICASSO-rec + pilot + delay: NMSE 0.84545512 +/- 0.01821252; delta +0.00033546; neutral
- PICASSO-rec + pilot consistency: NMSE 0.84559451 +/- 0.01823412; delta +0.00047484; neutral
- PICASSO-rec + pilot + frequency: NMSE 0.84569255 +/- 0.01809452; delta +0.00057289; neutral
- PICASSO-rec + energy preservation: NMSE 0.84569906 +/- 0.01895132; delta +0.00057939; neutral

- Pilot consistency: neutral (delta +0.00047484)
- Frequency consistency: neutral (delta +0.00008781)
- Delay sparsity: neutral (delta -0.00013362)
- Energy preservation: neutral (delta +0.00057939)
- Best physics variant: PICASSO-rec-full physics
- Full physics: neutral (delta -0.00019056). This aggregate result is neutral, so it shows neither a meaningful NMSE gain nor threshold-level over-regularization under this setting.

## 6. GAN Ablation

- PICASSO-rec without GAN: NMSE 0.84511966 +/- 0.01894520; delta +0.00000000; neutral
- PICASSO-rec + light adversarial loss: NMSE 0.84586543 +/- 0.01893124; delta +0.00074577; neutral
- PICASSO-rec + feature-level GAN: NMSE 0.96706723 +/- 0.13126827; delta +0.12194757; negative
- PICASSO-rec + output-level GAN: NMSE 1.13889048 +/- 0.05467614; delta +0.29377082; negative

- Output GAN: negative (delta +0.29377082)
- Feature GAN: negative (delta +0.12194757)
- Light adversarial: neutral (delta +0.00074577)
- Best adversarial variant: PICASSO-rec + light adversarial loss (neutral, delta +0.00074577).

## 7. Final Model Decision

PICASSO-rec remains the final model family. The best controlled structural configuration is `PICASSO-rec + refinement + multi-scale + SE` (positive, delta -0.00248651) and is the architecture recommended for the paper setting. The variant named `PICASSO-rec-enhanced full architecture` includes FiLM and is neutral (delta +0.00004744), so that full bundle should not replace the main model. Physics remains a secondary ablation. GAN remains deprecated unless a GAN variant has a positive contribution; observed decision: keep GAN deprecated.

## 8. Paper-Ready Interpretation

Architecture modules change reconstruction capacity directly and are evaluated without changing the objective. Physics terms can improve constraint satisfaction but may trade off against channel reconstruction when their priors mismatch CDL realizations. GAN objectives add optimization variance and are retained only when matched NMSE improves. Because all rows share sparse pilots, SNR, mobility, and profile coverage, the deltas isolate module contribution more credibly than cross-stage comparisons.

## 9. Limitations

This is a reduced but strict controlled study at one pilot ratio, SNR, velocity, and pilot pattern. It does not represent every possible wireless condition, and the simulator is CDL-inspired rather than a complete 3GPP implementation. Three profiles and three seeds improve coverage but do not establish universal statistical significance. The design is nevertheless sufficient to isolate module contributions within the stated setting.
