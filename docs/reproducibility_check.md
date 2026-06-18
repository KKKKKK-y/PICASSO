# Reproducibility Check

Date: 2026-06-18

## Policy

This was a minimal smoke validation only. No full Stage experiment, long training run, checkpoint save, or data dump was executed.

## Environment Summary

- Python: 3.11.15
- PyTorch: 2.11.0+cu128
- CUDA available: true
- Device used: cuda
- GPU: NVIDIA GeForce RTX 5070 Ti
- Repository path: `D:\SOTA\PICASSO`

## Smoke Procedure

The smoke check performed:

- synthetic dataset construction
- CDL dataset construction
- batch assembly for `H_sparse`, `H_full`, `mask`, and `H_prev`
- PICASSO-rec forward pass
- one optimizer step with `loss_mode="rec_physics"`
- LS and Enhanced-DnCNN forward checks
- metric computation for NMSE, MSE, MAE, pilot consistency, delay-domain sparsity, and Doppler robustness

## Shape Validation

- `H_sparse`: `[4, 2, 4, 4, 64]`
- `H_full`: `[4, 2, 4, 4, 64]`
- `H_hat`: `[4, 2, 4, 4, 64]`
- Synthetic dataset length: 4
- CDL dataset length: 4

## Runtime And Loss Sanity

- Runtime: 1.379 seconds
- Total loss: 0.4888574182987213
- Loss finite: true

## Metric Sanity

- PICASSO-rec smoke NMSE: 0.7840430736541748
- Enhanced-DnCNN smoke NMSE: 0.776954174041748
- LS smoke NMSE: 0.7744258642196655
- PICASSO-rec smoke MSE: 0.3920215368270874
- PICASSO-rec smoke MAE: 0.4826377332210541
- Pilot consistency error: 0.008883059024810791
- Delay-domain sparsity score: 0.35562437772750854
- Doppler robustness metric: 0.7840430736541748

## Conclusion

The minimal release smoke check passed. Dataset loading, model forward execution, one-step training, CUDA execution, tensor shapes, and loss/metric finiteness are verified.
