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
