# Bayesian fitting for rounded TTE data

`rounded_tte_bayesian_gof` fits event times observed as intervals and can run
the randomized posterior CDF diagnostic. It addresses the BCSTTE guide's
rounded survival-time example, where an integer `t` represents
`(t - 1/2, t + 1/2)`. The caller supplies interval endpoints directly; this
API does not fit right-censored data.

```python
import numpy as np
from mdanderson_stats import rounded_tte_bayesian_gof

rounded = np.array([1, 2, 2, 4, 5, 7])
lower = np.maximum(0, rounded - 0.5)
upper = rounded + 0.5

# Exponential model: Gaussian prior on absolute log-scale (reciprocal rate).
fit = rounded_tte_bayesian_gof(
    lower,
    upper,
    family="exponential",
    prior_mean=[np.log(3.0)],
    prior_covariance=[[0.5]],
    draws=100,
    warmup=100,
    chains=2,
    rng=np.random.default_rng(20261004),
)
assert fit.parameters.shape == (2, 100, 1)
assert fit.diagnostic is not None
print(fit.diagnostic.mean_statistic)
```

## Model and prior contract

The supported families are `exponential`, `weibull`, `lognormal`, `gamma`,
`inverse_gamma`, `log_logistic`, and `log_odds_rate`. For each, pass a proper
multivariate Gaussian prior on transformed absolute coordinates:
log-scale (the reciprocal of the rate) for exponential; log-shape/log-scale for Weibull,
Gamma, inverse-Gamma, and log-logistic; log-location/log-sigma for lognormal;
and log-shape/log-scale/log-c for log-odds-rate. The lognormal coordinates are
`(mu, log(sigma))`, where `mu` is the mean of log-time and may be negative.
`prior_mean` uses these absolute family coordinates; `prior_covariance` uses
the same axes and must be symmetric positive definite. Returned
`parameter_names` label posterior axes, with time-scale/location coordinates
centered as described below. These are
explicit Python prior choices and do not claim parity with the family-specific
conjugate priors of other APIs or unknown BCSTTE defaults.

For each row, the likelihood contribution is the probability that a draw from
the fitted model falls in `[lower, upper]`. Elliptical slice sampling retains
joint parameter draws. `parameters` uses chain/draw/parameter axes. The scale
or location coordinate is centered by `time_offset` for numerical stability;
the exponential log-scale uses the same subtraction convention. For lognormal,
the first coordinate is centered `mu` (the mean of log-time) and the second
remains `log(sigma)`.
The supplied prior mean is absolute; covariance is unaffected by centering.
Returned intervals, priors, resolved absolute initial coordinates, sampler
settings, likelihood work, and optional diagnostic are preserved in the result.
Use `compute_diagnostic=False` when
the fit is useful but floating-point CDF endpoints collapse for the rounded
intervals.

The diagnostic draws a uniform CDF position between each interval's paired
posterior CDF endpoints, then applies equal-probability bins. It is a
randomized discrete/rounded diagnostic, not a calibrated finite-sample p-value.
If a positive interval mass cannot be represented by distinct CDF floats, the
diagnostic raises rather than fabricating one. The fit may still be returned
without the diagnostic by explicitly disabling it.

Likelihood interval masses are computed from log-CDF or log-survival
differences. When both tails have endpoint separation no larger than
`32 * machine_epsilon * max(1, endpoint_log_magnitudes)`, the likelihood helper
rejects the unresolved interval instead of substituting a density-times-width
approximation. A probability below floating-point representability can also
produce zero representable mass and invalidate a chain start.

Bounds are finite and satisfy `0 <= lower < upper`; positive endpoints that
collapse after log-time centering are rejected. Right censoring, rounding-rule
inference, native priors, native random streams, and native report parity are
outside this API. Posterior draws are correlated: inspect chain summaries and
use adequate warmup, draws, and dispersed starts. The Johnson chi-square
reference remains asymptotic and subject to its regularity conditions.

Input evaluation is capped at one million draw-observation cells, retained
parameter arrays at two million cells, and diagnostic count arrays at two
million cells. Work and memory checks run before sampling. See the
[source-to-method audit](../research/rounded-tte-bayesian-gof-audit.md).
