# mTPI common Beta-prior sensitivity

The mTPI paper's Table 3 compares common independent `Beta(a,b)` dose priors
while leaving the equivalence interval and the penalties calibrated under
`Beta(1,1)` unchanged. This is prior sensitivity under the fixed mTPI loss
rule; it does not recalibrate losses for the alternative prior.

```python
import numpy as np
from mdanderson_stats import MTPIDesign, simulate_mtpi

design = MTPIDesign(
    target=0.30,
    lower=0.25,
    upper=0.35,
    prior_alpha=1.0,
    prior_beta=3.0,
)
posterior = design.posterior([3, 3], [1, 2])
decision = design.next_dose([3, 3], [1, 2], current_dose=2)
simulation = simulate_mtpi(
    design,
    [0.15, 0.30, 0.45],
    cohorts=4,
    cohort_size=2,
    trials=500,
    rng=np.random.default_rng(148),
)
print(posterior.probability, posterior.overdose_probability)
print(decision.action, simulation.selection_probability)
```

Each dose has an independent common prior `Beta(prior_alpha, prior_beta)`;
both shape parameters must be at least `1e-6`, with sum at most `1e6`. These
bounded numerical limits cover the paper's Table 3 priors. The default `(1,1)`
keeps the uniform-prior design and existing behavior. At dose `j`, the posterior is
`Beta(prior_alpha + y[j], prior_beta + n[j] - y[j])`. The three posterior
probabilities are divided by the unchanged interval widths, and the largest
UPM determines escalation, staying or de-escalation. The overdose safety
probability uses this same posterior and unchanged target/cutoff rule.

The final isotonic posterior mean and the optional isotonic posterior interval
calculation also use this prior. Priors are common across doses; dose-specific
priors and prior-dependent loss recalibration are outside this API. This is the
fixed-loss Table-3 sensitivity setup, not original TPI's separate calibrated
`K1`/`K2` method.

The paper does not specify every prior/interval combination or native random
stream. Alternative prior settings can change trial operating characteristics;
review the posterior probabilities and simulate the exact configuration of
interest. See the [source and validation audit](../research/mtpi-prior-sensitivity-audit.md).
