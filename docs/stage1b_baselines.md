# Stage 1B Baseline Evaluation

## Goal

Stage 1B expands the smoke-level sparse-pilot CSI reconstruction suite with
classical and neural baselines. It is still a verification stage, not a formal
training or benchmark run.

## Implemented Baselines

- `ls_baseline`: returns sparse pilot observations directly.
- `lmmse_like_baseline`: nearest-pilot interpolation with optional smoothing.
- `omp_like_baseline`: delay-domain sparse reconstruction using strongest taps.
- `CNNBaseline`: lightweight Conv2d smoke baseline from Stage 1A.
- `DnCNNBaseline`: residual Conv2d smoke baseline.

## Classical Approximation Notes

The LMMSE-like estimator does not use real channel covariance matrices. It is a
small interpolation and smoothing baseline that preserves observed pilots.

The OMP-like estimator is not a full iterative OMP solver. It interpolates
frequency-domain pilots, transforms to the delay domain, keeps the strongest
`n_paths` taps, and transforms back to the frequency domain.

## Neural Smoke Baselines

CNN and DnCNN are trained only for smoke validation. The default configuration
uses at most 256 synthetic samples and at most 2 neural epochs. No checkpoints
are saved.

## Metrics

The evaluation suite reports:

- NMSE
- MSE
- MAE
- pilot consistency error
- delay-domain sparsity score

The delay-domain sparsity score measures how much delay-domain energy is
concentrated in the strongest taps. It is a diagnostic score, not a standalone
paper metric.

## Artifact Policy

Stage 1B must not save generated datasets, checkpoints, NumPy arrays, model
outputs, or compressed artifacts. Results are printed to the terminal only.

## Run Command

```powershell
conda run -n picasso python src/picasso_csi/training/evaluate_baselines.py --config configs/stage1b_baselines.yaml
```

If the package is installed in editable mode, the module form is also available:

```powershell
conda run -n picasso python -m picasso_csi.training.evaluate_baselines --config configs/stage1b_baselines.yaml
```

## Stage 1C Plan

- Add stronger CNN and DnCNN variants with controlled training budgets.
- Add more realistic noise and SNR-aware pilot observations.
- Add a small comparison harness for multiple pilot ratios.
- Add PICASSO generator and discriminator skeletons without full GAN training.
- Define the first formal small-scale experiment protocol.
