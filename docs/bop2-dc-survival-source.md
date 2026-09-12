# Time-to-event BOP2-DC source notes

`bop2_dc_survival_design` implements the time-to-event monitoring model in
Section 2.1.3 of Zhao, Li, Liu, and Yuan, “Bayesian optimal phase II designs
with dual-criterion decision making” ([arXiv:2112.10880](https://arxiv.org/abs/2112.10880)).

The event time is exponential with mean `theta`, and the prior is explicitly
`InverseGamma(prior_shape, prior_scale)` on that mean. With `d` events and
total observed time `t`, the posterior is
`InverseGamma(prior_shape + d, prior_scale + t)`. The reported median scale is
`log(2) * theta`; consequently the posterior probability that median survival
exceeds threshold `m` is
`gammainc(a + d, (b + t) * log(2) / m)`.
The implementation forms this ratio from normalized prior and observed-time
contributions to preserve time-unit invariance for large finite scales.

Interim no-go requires both posterior tails to be below their scheduled
cutoffs. At the final look, both tails must exceed their controls for go, both
must be below them for no-go, and equality or mixed results produce consider.
This module accepts sufficient statistics and does not implement simulation or
operating-characteristic calibration.
