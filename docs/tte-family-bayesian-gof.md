# Additional BCSTTE family posterior fits

This module implements posterior workflows for the Gamma, inverse-Gamma, and
log-logistic time-to-event families defined in sections 4.2–4.4 of the cached
[BCSTTE user guide](bayesian-chi-square-sources.json). It uses the guide's
parameterizations: Gamma shape/scale `(alpha, beta)`, inverse-Gamma
shape/scale `(alpha, beta)`, and log-logistic `(k, theta)`, where the equivalent
shape/scale form has `scale = exp(-theta / k)`. Thus the log-logistic survival
function is `1 / (1 + (time / scale)**shape)`.

`gamma_bayesian_gof`, `inverse_gamma_bayesian_gof`, and
`log_logistic_bayesian_gof` take nonnegative observation times and require an
explicit proper bivariate Gaussian prior on `(log_shape, log_scale)`. This is
a caller-selected Python prior: the guide does not specify a prior or a
posterior-fitting algorithm. Parameter draws retain the joint shape/scale
pairing used to evaluate each observation's CDF.

For optional right-censoring, pass `event`, a Boolean vector with `True` for an
exact event and `False` for a right-censored time. Exact events use the family
density and censored observations use its right-tail survival function in the
likelihood. Censored times may be zero (their survival contribution is one),
but exact event times must be positive. No left truncation or rounded-time
likelihood is implied. The right-censor likelihood assumes noninformative
censoring and models only event times; it does not model the censoring
mechanism. Since the guide does not define a censored-data
probability-integral transform for the Johnson test, `diagnostic` is `None`
whenever any observation is censored. With complete data, the ordinary
posterior-CDF Johnson diagnostic is returned.

The second retained parameter and its chain summary are centered log scale;
add `log_scale_offset` to recover absolute log scale. Prior means and supplied
initial values use absolute log scale. To change time units by a factor `c`,
multiply times by `c` and add `log(c)` to the prior and initial log-scale
coordinates, keeping the covariance unchanged. Reported log likelihoods use
absolute time units. Split R-hat and batch-means Monte Carlo errors can help inspect
sampling, but do not guarantee convergence.

The log likelihood and Gamma tails are evaluated in centered log-time
coordinates. Underflowed incomplete-gamma tails use convergent series or
continued fractions in log space. Numerical failures and exhausted work
budgets raise errors; no partial fit is returned after a sampling failure.
Sampling uses serial elliptical slice updates with work, draw, and retained
array limits. These workflows do not claim native BCSTTE fitting, prior,
report, or executable parity.

```python
import numpy as np
from mdanderson_stats import gamma_bayesian_gof

fit = gamma_bayesian_gof(
    [1.2, 2.0, 3.1, 4.0],
    prior_mean=[np.log(2.0), np.log(2.0)],
    prior_covariance=[[0.5, 0.0], [0.0, 0.5]],
    rng=np.random.default_rng(17),
)
assert fit.diagnostic is not None
```
