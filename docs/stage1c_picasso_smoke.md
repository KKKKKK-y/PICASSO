# Stage 1C PICASSO Smoke Skeleton

## Goal

Stage 1C creates the first minimal PICASSO model loop. It adds generator and
discriminator skeletons, adversarial loss, composite physics-informed generator
loss, smoke training, and pilot-ratio smoke evaluation.

This stage is not full GAN training and does not produce formal experiment
results.

## Generator Skeleton

`PICASSOGenerator` is a lightweight residual Conv2d network. It accepts sparse
CSI tensors in channel-first format:

`(batch, 2, n_rx, n_tx, n_subcarriers)`.

Internally it reshapes the antenna dimensions into a 2D grid
`(n_rx * n_tx, n_subcarriers)`, predicts a residual, and returns `H_sparse +
residual` with the same shape as the input.

## Discriminator Skeleton

`PICASSODiscriminator` is a small CNN discriminator with global average pooling
and one real/fake logit per sample. It uses the same channel-first CSI format as
the generator.

## Loss Composition

The generator loss is:

`L_total = lambda_rec L_rec + lambda_adv L_adv + lambda_pilot L_pilot + lambda_smooth L_smooth + lambda_sparse L_sparse`

It combines reconstruction loss, BCE generator adversarial loss, Stage 0 pilot
consistency loss, Stage 0 frequency smoothness loss, and Stage 0 delay-domain
sparsity loss. Channel-first tensors are converted internally before calling
Stage 0 physics losses.

## Smoke Training Protocol

Run:

```powershell
conda run -n picasso python src/picasso_csi/training/smoke_train_picasso.py --config configs/stage1c_picasso_smoke.yaml
```

The default smoke config uses at most 256 generated samples and 2 epochs. The
script prints generator loss, discriminator loss, validation NMSE, and pilot
consistency error.

## Multi Pilot Ratio Evaluation

Run:

```powershell
conda run -n picasso python src/picasso_csi/training/evaluate_pilot_ratios.py --config configs/stage1c_picasso_smoke.yaml
```

The smoke evaluation prints terminal-only NMSE results for pilot ratios 0.5,
0.25, 0.125, and 0.0625 for LS, DnCNN, and PICASSO generator baselines.

## Artifact Policy

Stage 1C must not save checkpoints, generated data, NumPy arrays, model outputs,
or compressed artifacts. All results are printed to the terminal only.

## Limitations

The generator and discriminator are intentionally small skeletons. The
adversarial run is a smoke test for code integration, not a tuned GAN. Pilot
ratio results are sanity checks only.

## Stage 2 Plan

- Define a controlled small-scale experiment protocol.
- Add noise/SNR-aware sparse pilot observations.
- Tune neural baselines under fixed compute budgets.
- Expand PICASSO architecture only after baseline comparisons are stable.
- Begin limited GAN experiments with explicit checkpoint and artifact rules.
