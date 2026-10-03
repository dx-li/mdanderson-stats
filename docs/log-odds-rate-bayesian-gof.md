# Log-odds-rate survival posterior and Johnson diagnostic

`log_odds_rate_bayesian_gof` fits the generalized log-odds-rate survival family
to complete or right-censored times. It uses the Shen–Thall parameterization

```text
S(t) = [1 + c * (t / scale)**shape]**(-1 / c),
shape > 0, scale > 0, c > 0.
```

The family includes log-logistic survival at `c=1` and converges to Weibull
survival `exp(-(t/scale)**shape)` as `c` approaches zero. The guide does not
provide an executable prior or fitting algorithm. This Python workflow
therefore requires a caller-supplied proper correlated Gaussian prior on
`(log_shape, log_scale, log_c)`; it does not claim native fitting or prior
parity.

```python
import numpy as np
from mdanderson_stats.tte_family_bayesian_gof import log_odds_rate_bayesian_gof

fit = log_odds_rate_bayesian_gof(
    [0.45, 0.8, 1.3, 2.1, 3.4],
    prior_mean=np.log([1.6, 1.2, 0.7]),
    prior_covariance=[
        [0.13, 0.035, -0.018],
        [0.035, 0.19, 0.04],
        [-0.018, 0.04, 0.16],
    ],
    draws=1000,
    warmup=500,
    chains=4,
    rng=np.random.default_rng(771),
)
shape = np.exp(fit.parameters[..., 0])
scale = np.exp(fit.parameters[..., 1] + fit.log_scale_offset)
c = np.exp(fit.parameters[..., 2])
```

Set `event=True` for an exact event and `False` for a right-censored time.
Exact event times must be positive; a zero-time censor contributes survival one.
For censored input, `diagnostic` is `None` because no censored-data Johnson
transform is specified by the source. Complete observations retain paired
shape, scale, and `c` posterior draws when evaluating their CDFs and the
Johnson diagnostic.

The sampled log scale is centered on the log of the first positive observation
for numerical stability. The second coordinate of `parameters` is therefore
centered: add `log_scale_offset` to recover the absolute log scale. Supply
`prior_mean` and `initial` on absolute coordinates. If time units change by
factor `u`, multiply times by `u` and add `log(u)` to the prior and initial
log-scale coordinates; the covariance is unchanged. Reported log likelihoods
include the time-density Jacobian for each exact event and no such adjustment
for right-censored rows.

The sampler uses serial elliptical slice updates and validates covariance,
draw/work budgets, and retained memory before sampling. Posterior summaries
are diagnostics rather than convergence guarantees. See the
[source and numerical audit](../research/log-odds-rate-bayesian-gof-audit.md)
for equations and independent reference coverage.
