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

## Dependent order-statistic bounds

`diagnostic.order_bounds(upper_trim=.005)` or
`chi_square_order_bounds(statistics, degrees_of_freedom, upper_trim=.005)`
returns bounds for ascending ranks r of J posterior statistics:

```text
P(D_(r) > t) <= min(1, J * chi_square_sf(t) / (J - r + 1)).
```

The calculation follows equation (6) in [Yuan and Johnson (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3276744/).
It allows dependence between draws, provided their marginal reference holds.
The minimum across ranks is exposed as `minimum_diagnostic`; searching ranks
requires additional calibration. `search_adjusted_bound` conservatively multiplies
the minimum by the number of retained ranks, capped at one. Its validity also
requires the marginal reference, which is asymptotic for Johnson's chi-square
statistic. Exact pivotal arguments require the paper's prior assumptions;
an arbitrary improper prior does not establish them.

By default the largest `ceil(.005 * J)` observations are excluded from the search,
retaining at least one rank. Set `upper_trim=0` to retain all ranks. This follows
the later paper's tail-exclusion approach; BCSTTE's exact trimming and rank
conventions are unverified. Survival probabilities are floored at the smallest
positive normal float to avoid zero bounds from extreme-tail underflow.

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

Six focused tests cover independent Pearson calculations and endpoint bins,
reference summaries, exact Gamma posterior moments, seeded unit invariance over
400 orders of magnitude, repeated-data/posterior reference calibration, and a
clear nonexponential alternative. The repeated-data check uses independent
simulated datasets followed by one posterior draw per dataset; it does not treat
multiple posterior draws from a single dataset as independent calibration data.
The order-bound checks cover the published 20% tail example, strongly dependent
chi-square marginals, search correction, and extreme-tail underflow.

## Remaining coverage and source issues

**Catalog status is partial.** Right-censoring, rounded observations, the six
other distribution-family workflows, native fitting/priors and fallback priors,
native Rychlik rank/trim conventions, BIC/DIC, sorting and native HTML reports
remain pending.
The generic interface can consume verified posterior CDF draws from other models,
but it does not itself fit those models or impute censored observations.

The guide's `--censor` example conflicts with its option definition. Its log-logistic
variance omits subtraction of the squared mean, and its log-odds-rate survival
expression lacks the factor of c needed for the stated Weibull limit and moments.
These are recorded for the remaining source audit; they have not been implemented
as distribution definitions. No complete BCSTTE or censoring-method parity is claimed.
