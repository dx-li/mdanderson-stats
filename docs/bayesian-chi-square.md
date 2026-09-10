# Bayesian Chi Square TTE Fit: posterior diagnostic core

This implementation provides Johnson's posterior chi-square calculation for
complete continuous observations and an exponential workflow with an exact
Gamma posterior. The sources are the [BCS TTE guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BCSTTE/BCSTTE_UsersGuide.pdf)
(August 15, 2006) and Johnson's
[A Bayesian chi-square test for goodness-of-fit](https://arxiv.org/abs/math/0508593),
*Annals of Statistics* 32:2361–2384 (2004), equations (2)–(3).
[Provenance](bayesian-chi-square-sources.json) records the retrieved documents.
Original programs and documents are not redistributed.

## Generic posterior calculation

`bayesian_chi_square_cdf(cdf_samples, bins=None, critical_probability=.95)`
accepts a matrix with posterior draws as rows and observed data as columns.
Each row must evaluate the same observed dataset under one jointly sampled
parameter vector from its posterior. Covariate-specific distributions may vary
across observations, as in the paper's first corollary. Posterior-predictive
observations and repeated maximum-likelihood estimates are not substitutes.

Equal-probability intervals include their upper endpoints. Numerical CDF zero
belongs to the first interval. Counts produce the Pearson statistic with
expected count `n/K`, and the reference distribution has `K-1` degrees of freedom,
without subtracting fitted parameters. The default is nearest-integer `n**.4`,
with a minimum of two bins to keep positive degrees of freedom; the guide does
not specify this minimum. At least two observations are required.

The result retains bin counts, statistics, per-draw chi-square upper-tail
probabilities, the mean statistic, and the proportion exceeding a specified
reference critical value. `area_against_reference` is the posterior average of
the reference CDF, the paper's A diagnostic. `mean_reference_tail` is calculated
directly from upper tails. These averages are **not calibrated p-values** for the
posterior mean statistic. The chi-square reference is asymptotic under the paper's
regularity conditions; it does not make dependent posterior statistics independent.

## Exact exponential posterior

`exponential_bayesian_gof(times, prior_shape=..., prior_rate=..., samples=1000, ...)`
requires complete positive event times. The exponential rate has a Gamma prior
with explicitly supplied shape and rate. Posterior shape is prior shape plus n;
posterior rate is prior rate plus the sum of event times. Both prior parameters
may be zero, denoting the improper inverse-rate prior with a proper posterior
for the supported data. These priors are explicit Python choices, not recovered
BCSTTE defaults. Parameter samples are retained as log rates. Time sums and CDF
evaluation use log-space calculations to preserve unit invariance.

```python
import numpy as np
from mdanderson_stats import bayesian_chi_square_cdf, exponential_bayesian_gof

result = bayesian_chi_square_cdf([[0, 0.25, 0.5, 0.75, 1]], bins=4)
np.testing.assert_array_equal(result.bin_counts, [[2, 1, 1, 1]])
np.testing.assert_allclose(result.statistic, [0.6])
fit = exponential_bayesian_gof([1, 2, 4, 8], prior_shape=2, prior_rate=3, samples=1000, rng=66)
assert fit.posterior_shape == 6
np.testing.assert_allclose(fit.log_posterior_rate, np.log(18))
assert fit.diagnostic.statistic.shape == (1000,)
```

Three focused tests cover independent Pearson calculations and endpoint bins,
reference summaries, exact Gamma posterior moments, seeded unit invariance over
400 orders of magnitude, repeated-data/posterior reference calibration, and a
clear nonexponential alternative. The repeated-data check uses independent
simulated datasets followed by one posterior draw per dataset; it does not treat
multiple posterior draws from a single dataset as independent calibration data.

## Remaining coverage and source issues

**Catalog status is partial.** Right-censoring, rounded observations, the six
other distribution-family workflows, native fitting/priors and fallback priors,
the Rychlik p-value bound, sorting and native HTML reports remain pending.
The generic interface can consume verified posterior CDF draws from other models,
but it does not itself fit those models or impute censored observations.

The guide's `--censor` example conflicts with its option definition. Its log-logistic
variance omits subtraction of the squared mean, and its log-odds-rate survival
expression lacks the factor of c needed for the stated Weibull limit and moments.
These are recorded for the remaining source audit; they have not been implemented
as distribution definitions. No complete BCSTTE or censoring-method parity is claimed.
