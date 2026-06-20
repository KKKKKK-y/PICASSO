# PICASSO-CSI

Physics-Informed Channel Synthesis from Sparse Pilot Observations

PICASSO-CSI is a lightweight research codebase for sparse-pilot CSI reconstruction in OFDM and MIMO-OFDM systems. The project studies whether supervised and physics-guided neural reconstruction can recover full channel state information from sparse pilot observations under synthetic and CDL-inspired realistic wireless channels.

## Final Release Status

Stages 0-4 are complete and traceable. Stage 5A organizes their evidence into a paper-level ablation study and adds only reduced component-isolation runs. Stage 5B adds a strict matched-condition CDL ablation in which every variant shares the same data protocol, optimizer, batch size, epochs, and evaluator. The final default model is **PICASSO-rec**, a supervised residual reconstruction model. **PICASSO-rec-physics** is kept as the secondary physics-guided variant. **PICASSO-full / condition-aware full / GAN variants** are retained only as experimental ablations because adversarial training did not provide stable or consistent gains in the completed diagnostics.

The final evidence supports supervised PICASSO reconstruction as the primary path. Physics constraints are useful for analysis and consistency checks, but their measured gain is limited after Stage 3B/4. GAN training is deprecated as a primary direction for this release.

The strict Stage 5B controlled study keeps **PICASSO-rec** as the final model family. Under its fixed CDL setting, refinement + multi-scale + SE is the best structural configuration, while the full bundle including FiLM is neutral. Physics variants remain neutral/secondary, and output/feature GAN variants remain negative; light adversarial training does not provide a positive NMSE contribution.

## Repository Structure

- `configs/`: stage-specific reproducible experiment configs.
- `data/`: notes/placeholders only; no raw datasets are committed.
- `docs/`: design notes, stage reports, final audit, and release summaries.
- `literature/`: lightweight survey spreadsheets used for topic selection.
- `src/picasso_csi/datasets/`: synthetic, noisy synthetic, and CDL-inspired datasets.
- `src/picasso_csi/models/`: LS/LMMSE/OMP-style baselines, DnCNN variants, and PICASSO models.
- `src/picasso_csi/losses/`: reconstruction, physics, and optional legacy GAN losses.
- `src/picasso_csi/simulation/`: OFDM synthetic and CDL-inspired channel simulators plus pilot masks.
- `src/picasso_csi/training/`: stage runners and smoke-level validation entry points.
- `src/picasso_csi/evaluation/`: metrics and result-table helpers.
- `tests/`: unit and integration smoke tests.
- `outputs/results/`: committed lightweight CSV/Markdown result summaries only.

## Stage Trace

- **Stage 0-1A:** package skeleton, synthetic MIMO-OFDM channel generation, sparse pilot masks, datasets, physics losses, and smoke baselines.
- **Stage 1B:** LS, LMMSE-like, OMP-like, CNN, and DnCNN baseline evaluation.
- **Stage 1C:** PICASSO generator/discriminator skeletons, composite loss integration, and GAN smoke test.
- **Stage 2A:** noise-aware small formal evaluation across pilot ratios and SNR.
- **Stage 2B-2C:** diagnostic grid with random channel difficulty, condition-aware inputs, stronger baselines, ablations, and paper-style tables.
- **Stage 3A:** supervised physics-guided reconstruction before full GAN training.
- **Stage 3A-L:** larger controlled diagnostic including Enhanced-DnCNN, PICASSO-rec, PICASSO-rec-physics, PICASSO-full, and PICASSO-cond-full.
- **Stage 3B:** incremental structural enhancement and physics-loss ablation.
- **Stage 4:** CDL-inspired realistic wireless generalization with CDL-A/B/C, pilot pattern variation, pilot contamination, and mobility/Doppler diagnostics.
- **Stage 5A:** provenance-aware paper-level model, architecture, loss, GAN, channel-complexity, and robustness ablations with minimal gap-only runs.
- **Stage 5B:** new controlled CDL-A/B/C ablations at fixed pilot ratio, SNR, mobility, pilot pattern, seeds, and 30-epoch training budget.

## Recommended Usage

Run unit/smoke tests:

```powershell
conda run -n picasso python -m pytest
```

Run the final CDL Stage 4 protocol only when intentionally reproducing the Stage 4 experiment:

```powershell
conda run -n picasso python src/picasso_csi/training/run_stage4.py --config configs/stage4_cdl.yaml
```

Build the Stage 5A paper-level ablation tables. This reuses existing CSVs first and runs only the configured reduced component gaps:

```powershell
conda run -n picasso python src/picasso_csi/training/run_stage5a_ablation.py --config configs/stage5a_ablation.yaml
```

Run the strict Stage 5B controlled ablation:

```powershell
conda run -n picasso python src/picasso_csi/training/run_stage5b_controlled_ablation.py --config configs/stage5b_controlled_ablation.yaml
```

Earlier stage runners remain available for traceability:

```powershell
conda run -n picasso python src/picasso_csi/training/evaluate_baselines.py --config configs/stage1b_baselines.yaml
conda run -n picasso python src/picasso_csi/training/smoke_train_picasso.py --config configs/stage1c_picasso_smoke.yaml
conda run -n picasso python src/picasso_csi/training/run_stage2a_small_formal.py --config configs/stage2a_small_formal.yaml
conda run -n picasso python src/picasso_csi/training/run_stage2bc_comprehensive.py --config configs/stage2bc_comprehensive.yaml
conda run -n picasso python src/picasso_csi/training/run_stage3a_supervised_physics.py --config configs/stage3a_supervised_physics.yaml
conda run -n picasso python src/picasso_csi/training/run_stage3a_larger_training.py --config configs/stage3a_larger_training.yaml
conda run -n picasso python src/picasso_csi/training/run_stage3b_incremental.py --config configs/stage3b_incremental.yaml
```

## Artifact Policy

Do not commit raw datasets, checkpoints, NumPy/Mat/H5 dumps, prediction tensors, or large generated outputs. Local checkpoints under `checkpoints/` are ignored. The repository commits only source code, configs, documentation, tests, lightweight survey spreadsheets, and compact result CSV/Markdown files under `outputs/results/`.

## Final Recommendation

For paper writing and future extensions, use PICASSO-rec as the main method, Enhanced-DnCNN and LS/LMMSE-style estimators as fairness baselines, and PICASSO-rec-physics as the physics-consistency ablation. Treat full GAN variants as historical/optional ablations unless future realistic data provides stronger evidence for adversarial gains.
