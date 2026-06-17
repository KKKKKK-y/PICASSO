# Stage 1A Smoke Baseline

## Goal

Stage 1A validates the minimal sparse-pilot CSI reconstruction pipeline before
any full PICASSO-GAN training. The smoke path is:

MIMO-OFDM channel generation -> sparse pilot mask -> PyTorch Dataset/DataLoader
-> LS/CNN/DnCNN baselines -> MSE/MAE/NMSE metrics -> two-epoch smoke training.

This stage is not a formal experiment and should not be used for paper-level
performance claims.

## Dataset Shape

Stage 0 simulation utilities generate channels in real-imaginary format with
shape `(n_rx, n_tx, n_subcarriers, 2)`. `SyntheticCSIDataset` converts each
sample to PyTorch channel-first format:

`(2, n_rx, n_tx, n_subcarriers)`.

Each sample contains:

- `H_sparse`: sparse pilot observations
- `H_full`: full synthetic CSI target
- `mask`: pilot observation mask

All samples are generated in memory or on demand. The dataset does not write
`.npy`, `.npz`, `.mat`, `.h5`, or any other generated data file.

## Baseline Design

`LSInterpolationBaseline` performs nearest-pilot interpolation along the
frequency axis. It is a lightweight shape and sanity baseline, not a complete
least-squares channel estimator.

`CNNBaseline` concatenates `H_sparse` and `mask` into four input channels,
reshapes `(n_rx, n_tx, n_subcarriers)` into a 2D grid
`(n_rx * n_tx, n_subcarriers)`, and predicts two real-imaginary output channels.

`DnCNNBaseline` uses the same grid representation with residual learning:

`H_hat = H_sparse + residual`.

## Smoke Training Command

From the repository root:

```powershell
conda run -n picasso python -m src.picasso_csi.training.smoke_train_baseline --config configs/smoke.yaml
```

The script can also be run directly:

```powershell
conda run -n picasso python src/picasso_csi/training/smoke_train_baseline.py --config configs/smoke.yaml
```

The default smoke config uses 128 training samples, 32 validation samples,
batch size 16, and 2 epochs.

## Artifact Policy

Stage 1A must not save checkpoints, generated datasets, output tensors, or large
experiment artifacts. `configs/smoke.yaml` keeps all artifact saving flags set
to `false`.

## Stage 1B Plan

- Implement stronger CNN and DnCNN baselines.
- Add OMP and LMMSE baselines.
- Add a full synthetic dataset generation script with explicit artifact policy.
- Add PICASSO generator and discriminator skeletons.
- Start small-scale baseline comparison.
