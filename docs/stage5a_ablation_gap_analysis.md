# Stage 5A Ablation Gap Analysis

## Evidence Inventory

Stage 2BC supplies multi-seed synthetic model, condition, physics-on/off, and light adversarial comparisons. Stage 3A-L supplies the controlled larger-model comparison and parameter counts. Stage 3B isolates aggregate structural enhancement, SE attention, feature-level GAN, physics, and FiLM effects. Stage 4 supplies CDL-A/B/C, mobility, SNR, pilot-ratio, and comb/block/irregular-pilot evidence.

## Gap Decisions

1. **Model ablation:** sufficient for the principal model families after combining Stage 2BC, 3A-L, and 3B. The historical runs differ in budget, so cross-stage values must retain provenance.
2. **Loss ablation:** insufficient. Existing work compares reconstruction against bundled physics, but does not isolate pilot, frequency, delay, and energy terms.
3. **GAN ablation:** sufficient for the paper conclusion. Output-level/light GAN evidence from Stage 2BC/3A-L and feature-level evidence from Stage 3B consistently fail to justify GAN as the main path.
4. **Condition ablation:** sufficient as a negative/unstable result. Stage 2BC and Stage 3B both contain conditioned controls, with no stable overall benefit.
5. **Channel complexity:** sufficient for fixed-path noisy synthetic, noisy/random-path synthetic, and CDL-A/B/C. A simple noiseless Stage 1 row-level CSV and a strict standardized synthetic-to-CDL transfer experiment are absent, so comparisons are descriptive rather than causal; the old smoke experiment is not reconstructed from prose.
6. **Robustness:** sufficient for the requested pilot ratios, 0/10/30 dB SNR, 0/60/120 km/h, and three pilot patterns by reusing Stage 3B/4. The optional 0.05 pilot ratio is not needed.
7. **Directly reusable CSVs:** `stage2a_small_formal_results.csv`, `stage2bc_raw_results.csv`, `stage3a_larger_raw_results.csv`, `stage3b_incremental_results.csv`, and `stage4_raw_results.csv` are the primary row-level sources.
8. **Must rerun:** only isolated architecture components and isolated physics-loss components, on a representative reduced condition with two seeds.
9. **Not worth rerunning:** the full GAN grid, full Stage 4 grid, long Stage 3A-L training, and optional 0.05 pilot ratio. Existing evidence already answers those questions and rerunning would spend compute without changing the inference boundary.
10. **Paper tables to retain:** main model comparison, component ablation, and robustness comparison. Detailed per-profile/per-seed rows belong in supplementary material.

## Scientific Boundary

Historical rows are reused without alteration. New reduced-grid rows answer only component-isolation questions; they are not presented as replacements for the larger historical experiments. No cross-stage ranking is treated as a controlled head-to-head result unless the conditions and source stage match.
