# Fixed-shape Weibull Bayesian goodness of fit

`weibull_fixed_shape_bayesian_gof` extends the
[Bayesian Chi Square TTE diagnostic](bayesian-chi-square.md) to a Weibull
model with a caller-fixed positive shape and an explicit Gamma prior on its
transformed rate. It uses exact independent posterior draws, then evaluates
Johnson's statistic on the observed data under each draw.

```python
import numpy as np
from mdanderson_stats import weibull_fixed_shape_bayesian_gof

beta = 1.7
observed = np.array([0.5, 1.25, 2.0, 3.5])
fit = weibull_fixed_shape_bayesian_gof(
    observed,
    weibull_shape=beta,
    prior_shape=2.25,
    prior_rate=0.8,
    samples=2000,
    bins=3,
    rng=6607,
)
assert fit.posterior_shape == 6.25
np.testing.assert_allclose(
    fit.log_posterior_rate,
    np.log(0.8 + np.sum(observed**beta)),
)
assert fit.diagnostic.statistic.shape == (2000,)
# Ordinary-scale posterior log rates; see the extreme-scale representation below.
log_rates = fit.centered_log_rate_samples + fit.log_rate_offset
print(log_rates.mean(), fit.diagnostic.area_against_reference)
```

## Model and prior

For shape `beta`, scale `eta`, and transformed rate `lambda = eta**(-beta)`,

```text
F(t) = 1 - exp(-lambda * t**beta)
lambda ~ Gamma(prior_shape, rate=prior_rate)
lambda | observed times ~ Gamma(prior_shape + n, rate=prior_rate + sum(t**beta))
```

All observations must be complete positive continuous event times. Shape is
specified, not estimated. The prior rate has units `time**beta`. A change of
time units by factor `c` requires changing the prior rate by `c**beta` to
represent the same prior. Zero prior hyperparameters define improper priors;
the supported complete data yield a proper posterior. For example, setting
both to zero gives the inverse-rate prior. These are explicit Python choices,
not recovered BCSTTE prior defaults.

The shape-one model reduces to the exponential workflow. Other shapes permit
increasing or decreasing hazards. The function does not estimate an unknown
Weibull shape or supply censored/rounded-data diagnostics. It therefore adds a
specified-shape family without claiming complete coverage of the native
Weibull fitter.
The companion [unknown-shape workflow](weibull-unknown-shape-gof.md) jointly
estimates shape and scale with an explicit Gaussian prior in log coordinates.

## Numerical representation and interpretation

Power sums, posterior rates and CDF hazards are evaluated on a common scale.
Relative times are centered before multiplication by shape, preserving nearby
floating-point times even when absolute log powers are enormous. The result
retains two complementary representations:

- `centered_log_rate_samples` and `log_rate_offset` encode each posterior log
  rate as their sum. Keep them separate at extreme scales: adding a huge
  offset can erase the posterior variation.
- `posterior_rate_log_scale` and `posterior_rate_scaled_sum` encode the Gamma
  posterior rate as `exp(log_scale) * scaled_sum`. `log_posterior_rate` is a
  convenience scalar and may lose low-order precision at extreme scales.

Unrepresentable common log scales or Gamma draws raise a clear error. Storage
checks include retained CDF/count arrays, working vectors and chunk buffers
before draws or large conversion. CDFs are computed in bounded row chunks.
The limits bound computation and memory, not the precision of an estimate.

The returned `diagnostic` uses the existing equal-probability bin convention
and `bins-1` reference degrees of freedom. Its average reference probability
and exceedance fraction are posterior diagnostics, not calibrated p-values
for the posterior mean statistic. Johnson's chi-square reference is asymptotic;
posterior statistics from one dataset are not repeated independent datasets.

## Independent validation

The [base-R reference](../tools/reference_weibull_bayesian_gof.R) evaluates the
exact Gamma posterior and integrates diagnostic expectations over every
interval on which the observed-data bin counts are constant. Seven cases
cover shape one, increasing/decreasing hazards, time units changed by
`1e-200` and `1e200`, and a common log-power offset of order `1e307`.
All 76 checked posterior/CDF/diagnostic summaries agree within 1.671 estimated
Monte Carlo standard errors. Six focused checks also cover exponential
reduction, direct CDF identities, neighboring representable times at extreme
shape, and pre-RNG resource rejection. See the
[audit](../research/bayesian-chi-square-weibull-audit.md) for scope and evidence.
