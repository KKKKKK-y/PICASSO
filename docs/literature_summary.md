# Literature Summary

## Key Observations From Literature Survey

The existing survey shows that sparse CSI reconstruction, CSI feedback, RIS channel estimation, and near-field XL-MIMO estimation all suffer from pilot or feedback overhead. Among these directions, MIMO-OFDM sparse-pilot CSI reconstruction is the cleanest first target because the task can be defined directly as sparse pilot observations to full CSI.

## Most Suitable Scenario

The recommended first scenario is MIMO-OFDM sparse-pilot CSI reconstruction. It is suitable for IEEE Communications Letters because the problem is narrow, the system model is compact, the baselines are mature, and the physics-informed losses can be described clearly.

## Recommended Datasets/Simulators

Primary route:

- Self-built MIMO-OFDM simulation

Recommended extensions:

- 3GPP CDL-A
- 3GPP CDL-B
- 3GPP CDL-C
- QuaDRiGa

Secondary options:

- DeepMIMO for Massive MIMO or ray-tracing-based CSI
- COST2100 for Massive MIMO CSI feedback comparisons
- Raymobtime for mobility or ray-tracing studies

## Recommended Baselines

Minimum baseline set:

- LS
- LMMSE
- OMP
- CNN
- DnCNN
- GAN without physics
- PINN without GAN
- Proposed PICASSO

## Main Risks

- GAN training instability
- Unfair LMMSE covariance assumptions
- Weak baselines if only LS/CNN are compared
- Over-expanding the first paper into RIS, XL-MIMO, or CSI feedback
- Physics losses becoming generic smoothing rather than meaningful OFDM constraints

## V3 Large-Scale Survey Summary

V3 large-scale survey further confirms that the first PICASSO paper should not start from a broad “AI for wireless” topic, but from a narrow and reproducible sparse-pilot CSI reconstruction problem. The initial search collected a large candidate pool across low-pilot channel estimation, MIMO-OFDM reconstruction, GAN/diffusion-based channel generation, CSI feedback, model-driven deep learning, RIS channel estimation, and Massive MIMO recovery. After removing duplicate, weakly related, tutorial-like, off-topic, and pure resource-optimization papers, the final spreadsheet keeps 72 effective papers. More than 70% of the selected papers are from 2023–2026, and more than half are IEEE sources, so the survey is recent enough and aligned with the target publication ecosystem.

The strongest conclusion is that MIMO-OFDM sparse-pilot CSI reconstruction remains the most suitable first scenario. It gives a clean input-output mapping: low-pilot LS channel estimates plus pilot masks as input, full CSI over the OFDM grid as output. This mapping is easier to explain than CSI feedback, RIS cascaded estimation, or XL-MIMO near-field reconstruction. It also supports fully synthetic and reproducible experiments without downloading large private datasets. A self-built MIMO-OFDM simulator should be the primary route, with 3GPP CDL-A/B/C or QuaDRiGa as optional extensions once the minimum experiment is stable.

The survey also shows that GAN-based channel estimation and channel generation are already active but often lack explicit wireless physical constraints. Existing cGAN, SRGAN, WGAN, diffusion, and flow-matching papers demonstrate that generative models can reconstruct or synthesize CSI from noisy, quantized, incomplete, or compressed observations. However, many of them treat the channel mainly as an image-like tensor and do not enforce pilot consistency, OFDM structure, or delay-domain sparsity. This gap is exactly where PICASSO can position itself: the generator synthesizes full CSI from sparse pilot observations, while physics-informed losses constrain the generated channel to match pilot measurements, preserve time-frequency smoothness, and respect multipath sparsity.

Physics-related and model-driven papers provide the second pillar. Deep unfolding, sparse Bayesian learning, OMP, Kalman filtering, tensor decomposition, and message passing works all show that wireless channel estimation benefits from embedding observation equations and structural priors. These works do not necessarily use PINNs in the strict PDE sense, but they justify the broader physics-informed design. For PICASSO, the most defensible constraints are pilot consistency loss, delay-domain sparsity loss, time-frequency smoothness loss, and optional delay-domain consistency. These are easier for IEEE Communications Letters reviewers to accept than vague “physical regularization.”

For the first paper, the minimum baseline set should include LS, LMMSE, OMP, CNN, DnCNN, GAN without physics, PINN without GAN, and the proposed PICASSO. Diffusion and Transformer baselines are useful but may be too heavy for the first short paper unless implementation time allows. The key ablations should remove physics loss, adversarial loss, pilot consistency loss, and sparsity loss separately. The recommended title remains: “PICASSO: Physics-Informed Channel Synthesis from Sparse Pilot Observations.”
