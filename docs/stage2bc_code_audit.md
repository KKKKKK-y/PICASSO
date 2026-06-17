# Stage 2BC Code Audit

## Tensor Shapes

Current dataset samples use channel-first real/imaginary tensors:

- `H_full`: `(2, n_rx, n_tx, n_subcarriers)`
- `H_sparse`: `(2, n_rx, n_tx, n_subcarriers)`
- `mask`: `(2, n_rx, n_tx, n_subcarriers)`

DataLoader batches therefore use `(B, 2, n_rx, n_tx, n_subcarriers)`. CNN models flatten the antenna grid to `(B, C, n_rx * n_tx, n_subcarriers)` before applying Conv2d.

## Why Zero-Filled LS Is Strong

The current LS baseline returns the noisy sparse observation directly. NMSE can look strong because the unobserved entries are zeros and the synthetic channels are normalized, smooth, and delay sparse. At higher pilot ratios, a large fraction of useful energy is already preserved at observed pilots, so a network must learn interpolation without hurting pilot consistency.

## Noisy Sparse Observation Realism

Stage 2A added AWGN at pilot locations and zeros elsewhere. This is a useful pilot-observation abstraction, but it is not yet a full receiver pipeline with pilot symbol division, interpolation, carrier effects, or realistic channel families. Stage 2BC keeps clean `H_full` labels and pilot-only noisy `H_sparse`, while adding random path count and random delay spread.

## Why DnCNN Barely Beats LS

DnCNN receives only `H_sparse` and `mask` in Stage 2A. It is trained for a very small number of epochs on a simple synthetic channel, so it can easily learn a near-identity residual and avoid damaging observed pilots. Without SNR or pilot-ratio conditioning, one network cannot explicitly adapt its denoising/interpolation behavior across all operating points.

## Why PICASSO Has Not Beaten LS

The Stage 1C/2A generator is a skeleton residual CNN, and the discriminator sees only generated/real channel tensors. The adversarial loss can add instability before supervised reconstruction is strong. Physics losses may also encourage smooth or sparse channels, but if the weights are high relative to reconstruction, they can over-constrain the estimator.

## Generator Capacity

The original generator uses 32 base channels and 3 residual blocks. That is enough for integration tests but likely weak for low-pilot interpolation. Stage 2BC increases the diagnostic generator to 64 base channels and 4 residual blocks while keeping runtime bounded.

## Discriminator Role

The discriminator has limited practical effect in the current small-budget setting. It does not observe the mask, SNR, or pilot ratio, so it cannot judge whether a reconstruction is conditionally plausible. Stage 2BC keeps adversarial variants for diagnosis, but treats GAN evidence cautiously.

## Physics Loss Strength

Pilot consistency is physically meaningful. Frequency smoothness and delay sparsity are plausible for the current synthetic generator, but can be too strong under random delay spread or richer channels. Stage 2BC ablates reconstruction-only, reconstruction-plus-physics, reconstruction-plus-adversarial, and full loss modes.

## Synthetic Channel Difficulty

The original simulator used fixed path count and uniform delays. This can make the task too simple and favor LS or shallow residual denoisers. Stage 2BC adds random path counts, random delay spread, normalized channel power, and pilot-only AWGN to create more varied diagnostic conditions.

## Metrics

NMSE, MSE, MAE, pilot consistency error, and delay-domain sparsity score are reasonable for diagnostic experiments. NMSE remains the main accuracy metric. Pilot consistency error should be interpreted carefully because an LS-style estimator can score well by construction.

## Shape, Device, And Dtype Risks

The largest shape risk is switching between channel-first tensors for models and channel-last tensors for physics losses. Stage 2BC keeps model tensors channel-first and converts only inside `picasso_generator_loss`. Device risk is mostly condition channels, so the new conditioning helper explicitly accepts device and batch tensors.

## Condition Channels

Condition channels are needed. SNR and pilot ratio materially change the reconstruction problem, and a single unconditioned model is forced to infer these settings from sparse magnitudes and masks. Stage 2BC adds normalized SNR and pilot-ratio channels for Cond-DnCNN and PICASSO-cond-full.

## Data Generation Changes Needed

Stage 2BC extends the dataset with random path counts, random delay spread, complex Gaussian gains, channel normalization, pilot-only noise, clean labels, explicit pilot ratio, SNR, and deterministic seeds. More realistic Stage 3 work should still add 3GPP CDL or QuaDRiGa channels.
