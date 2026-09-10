# Bayesian updating for normal data (BNORM)

Catalog entry **103** is implemented from the official
[BNORM help file](https://biostatistics.mdanderson.org/shinyapps/BNORM/BNORM.pdf),
by Yanhong Zhou and J. Jack Lee. The Python API covers both documented conjugate
models, raw and summary data, sequential updating, prior sensitivity, marginal
distributions, credible intervals and data-only confidence intervals. The browser
controls and animated presentation are not reproduced; arrays of density values
are available for plotting. [Source provenance](normal-updating-source.json)
pins the inspected manual. This is an independent mathematical implementation,
not a claim to reproduce an unavailable Shiny server implementation.

## Known variance

```python
from mdanderson_stats import NormalSample, NormalMeanPosterior

sample = NormalSample.from_data([1, 2, 3, 4, 5])
prior = NormalMeanPosterior(location=0, variance=10)
posterior = prior.update(sample, observation_variance=4)
print(posterior.location, posterior.variance)  # 2.77777778, 0.74074074
print(posterior.credible_interval())  # approximately [1.09091, 4.46465]

# Summary input and repeated updating use the same statistical contract.
next_sample = NormalSample(mean=2, size=5)
next_posterior = posterior.update(next_sample, observation_variance=4)
```

Prior precision is 1/v0 and data precision is n/v. Posterior precision is their
sum; the posterior mean is their precision-weighted average. The implementation
combines precision in logarithms, avoiding overflow from reciprocal variances.
The result supplies `pdf`, `cdf`, `sf` and `credible_interval`. A symmetric normal
interval is both equal-tailed and highest density. `variance` describes the
uncertainty in the unknown mean, not the observation variance.

## Unknown variance

```python
from mdanderson_stats import NormalInverseGamma

prior = NormalInverseGamma(location=0, mean_precision=4, shape=2, scale=3)
posterior = prior.update(sample)
print(posterior.location, posterior.mean_precision, posterior.shape, posterior.scale)
# 1.66666667, 9, 4.5, 18
print(posterior.mean_interval())  # [0.15856189, 3.17477144]
print(posterior.variance_interval())  # [1.38053057, 10.96373767]
```

The parameterization is
`mu | v ~ Normal(location, v / mean_precision)` and
`v ~ InverseGamma(shape, scale)`, whose density is proportional to
`v**(-shape-1) * exp(-scale/v)`. In the manual's alternative parameterization,
`mean_precision=n0`, `shape=nu0/2`, and `scale=nu0*sigma0_squared/2`.

For data mean m, size n and centered sum of squares S, updating gives

```text
k_new = k + n
location_new = (k * location + n * m) / (k + n)
a_new = a + n / 2
b_new = b + S / 2 + (k * n / (k + n)) * (location - m)**2 / 2
```

The scale accumulation uses logarithms to avoid intermediate square overflow.
Mean calculations preserve small differences around large offsets and fall back
to scaled arithmetic for overflowing differences. Marginally, mu is Student t
with `df=2*shape`, location `location` and scale `mean_scale`, where
`mean_scale=sqrt(scale/(shape*mean_precision))`. That scale is **not** the Student-t
standard deviation. `mean_expectation` is NaN when it does not exist;
`variance_expectation` is infinite for shape <= 1.

The `mean_pdf`, `mean_cdf`, `mean_sf` and `mean_interval` methods describe mu.
The corresponding `variance_*` methods describe v. `variance_interval` defaults
to highest density and accepts `method="equal-tail"`. Highest-density boundaries
are found by vectorized probability-space bisection at unit scale, then rescaled.
The solver checks both interval mass and equal boundary density.

## Data summaries and batching

```python
summary = NormalSample(mean=3, size=5, sample_sd=2.5**0.5)
updated = prior.update(summary)
data_only = summary.confidence_interval()
known_variance_data_only = summary.confidence_interval(observation_variance=4)
```

`sample_sd` uses the unbiased n-1 denominator, so S=(n-1)*sample_sd**2.
`NormalSample.from_data` reduces the last axis; leading axes represent independent
samples. Parameter and summary arrays broadcast for sensitivity analyses.
Results own their stored arrays and make them read-only. Repeated `update` calls
return new posterior objects and are equivalent to pooling independent data
under the same model.

Sample size must be a positive integer below 2**53. Missing sample spread is
allowed for known-variance summary input; unknown-variance updating rejects it
when n>1. A singleton has S=0 and undefined sample variance. Data-only intervals
use normal quantiles with known variance, or Student-t quantiles with sample
variance and n>=2. They are distinct from posterior credible intervals.

Proper priors require positive finite variance/precision/shape/scale. Invalid
statistics, unrepresentable sums of squares or posterior parameters, and failed
interval calculations raise errors. Extremely small spreads whose squares
underflow are rejected rather than interpreted as constant observations.
Intervals require probability in (0,1); unrepresentable endpoints are rejected.

## Manual discrepancy and validation

Page 13 labels (0.44, 3.08) for IG(2,3) and (1.87, 6.53) for IG(4.5,18) as 95%
variance intervals. Their masses are approximately 0.73676 and 0.76435,
respectively. This observation concerns the inspected manual, not an unverified
claim about the current live app. Python provides mass-checked intervals instead.
The manual's posterior parameters and marginal mean interval are reproduced.

Fourteen focused tests check those examples, pooled/sequential equivalence,
raw/summary equivalence, broadcasting, large-offset and unit-change invariance,
normal and Cauchy density identities, exact integer-shape inverse-gamma CDFs,
highest-density and equal-tail interval mass, data-only intervals, undefined
moments, extreme scales and invalid inputs. These tests validate the statistical
contracts without duplicating a separate test for every display helper.
