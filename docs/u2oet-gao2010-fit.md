# Original 2010 GAO posterior fitting

`fit_u2oet_gao2010` fits the original centered-dose ordinal GAO probability
model using explicit caller-supplied prior centers and scales. It is separate
from the 2017 GAO fitter. It does not reproduce the native application sampler
or infer the paper's elicited prior centers.

```python
import numpy as np

from mdanderson_stats import fit_u2oet_gao2010, u2oet_gao2010_parameter_names

dose_a, dose_b = [0.0, 1.0], [0.0, 2.0]
counts = np.zeros((2, 2, 2, 2), dtype=int)
counts[0, 0, 0, 0] = 2
counts[1, 1, 1, 0] = 1
names = u2oet_gao2010_parameter_names(2, 2)
# Supply one mean and standard deviation for each name except association.
prior_mean = np.zeros(len(names) - 1)
prior_sd = np.full(len(names) - 1, 0.4)
fit = fit_u2oet_gao2010(
    dose_a,
    dose_b,
    counts,
    prior_mean=prior_mean,
    prior_sd=prior_sd,
    draws=100,
    warmup=100,
    chains=2,
    rng=np.random.default_rng(20260929),
)
```

The named Gaussian coordinates are threshold-major within each endpoint:
intercept for agent 1, intercept for agent 2, slope for agent 1, slope for
agent 2; then endpoint `log_lambda` and `gamma`. Efficacy coordinates precede
toxicity coordinates. The final retained coordinate is raw copula association
in `[-1, 1]`; its default prior is Uniform(-1,1). Set
`fixed_association=...` for a conditional fit. Gaussian standard deviation zero
fixes that coordinate at its supplied mean. Dose grids and count axes follow
the fixed-parameter [2010 model guide](u2oet-gao2010.md).

For negative interaction values, the target density is zero unless the model
bracket is strictly positive at every supplied-grid dose pair and threshold.
The independent Gaussian coordinates are restricted by one joint validity
indicator; there is no conditional renormalization of gamma given the other
coordinates. This is an explicit Python prior convention. The paper reports
normal priors for coefficients/interactions, lognormal priors for lambda, and
uniform association; it does not establish the caller's native parameter-file
ordering or executable sampler settings. Its reported variances are 100 for
alpha and 2.25 for log(lambda) and gamma, but its elicited prior centers require
a separate calibration that is not implemented here.

The sampler uses endpoint-block elliptical slice updates for Gaussian
coordinates and an independent uniform Metropolis proposal for association.
The returned draws retain chain and draw axes; posterior arrays, the supplied
dose grids, counts, and prior vectors are read-only. The grids are retained
because they define the joint support restriction. The summary and acceptance
values are diagnostics, not proof of convergence.
Explicit `initial` values must be valid on the supplied dose grid. Without
them, chains start at the prior centers and association zero (or the fixed
association); choose dispersed valid starts when assessing multimodality.

The implementation preflights retained arrays, the minimum likelihood work,
and the minimum evaluations. Hard ceilings are 20 million retained scalar
cells, 100 million dose/category work units, and 1 million likelihood
evaluations. Elliptical slice updates also have a bounded 1,000-step bracket
search. Increase runtime cautiously; the Gaussian-copula rectangle kernel has
its own numerical work limit. No observations are generated or retained by
this fitter.
