# Normal-data hierarchy (BHM-NN)

Catalog entry **107** is implemented from the official
[BHM-NN manual](https://biostatistics.mdanderson.org/shinyapps/BHM-NN/BHM_NN.pdf),
by Yanhong Zhou and J. Jack Lee. The Python implementation provides the specified
three-level model, group and overall posterior draws, precision draws, and the
empirical group/pooled comparisons underlying the manual's descriptive plots.
[Provenance](hierarchical-normal-source.json) pins the inspected document.
Historical CSV templates, report layouts and JAGS random-number sequences are
not reproduced.

## Model

```text
y_ij | theta_i,phi_i ~ Normal(theta_i, 1/phi_i)
theta_i | mu,tau ~ Normal(mu, 1/tau)
phi_i ~ Gamma(observation_shape, rate=observation_rate)
mu ~ Normal(prior_mean, 1/prior_mean_precision)
tau ~ Gamma(between_shape, rate=between_rate)
```

The group-specific observation precisions phi_i differ from the shared
between-group precision tau. All Gamma parameters use shape/rate. The priors for
group means and group observation precisions are conditionally independent;
this is not the normal–inverse-gamma parameterization used in BNORM.

## Usage

```python
import numpy as np
from mdanderson_stats import hierarchical_normal, summarize_chains

groups = [
    [2, 6, 7, 3.8, 4.2],
    [11, 12, 10.7, 6.5, 7.9, 12.3, 15],
]
fit = hierarchical_normal(groups, draws=4000, warmup=1000, rng=np.random.default_rng(107))
group_summary = summarize_chains(fit.group_mean)
overall_summary = summarize_chains(fit.global_mean)
observation_diagnostics = summarize_chains(fit.observation_precision)
between_diagnostics = summarize_chains(fit.between_precision)

markers = np.linspace(-5, 20, 301)
group_descriptive_densities = fit.empirical_density(markers)
pooled_descriptive_density = fit.empirical_density(markers, pooled=True)
```

For summary input, pass a `NormalSample` containing vectors of group means,
positive sample sizes and unbiased sample standard deviations. Raw input is a
sequence of 2 through 100 nonempty one-dimensional groups. It uses the existing
stable normal sufficient-statistic implementation. Singleton groups are supported
by the hierarchy, with centered sum of squares zero. Missing spread for larger
groups is rejected. Total sample size must be below 2**53.

## Gibbs updates

For group size n_i, sample mean ybar_i and centered sum of squares S_i, each
iteration samples the following full conditionals:

```text
theta_i: Normal with precision n_i*phi_i + tau,
         mean (n_i*phi_i*ybar_i + tau*mu)/(n_i*phi_i + tau)
phi_i:   Gamma(observation_shape+n_i/2,
               rate=observation_rate+(S_i+n_i*(ybar_i-theta_i)**2)/2)
mu:      Normal with precision prior_mean_precision+m*tau,
         mean (prior_mean_precision*prior_mean+tau*sum(theta_i))/precision
tau:     Gamma(between_shape+m/2,
               rate=between_rate+sum((theta_i-mu)**2)/2)
```

Updates are vectorized across groups and chains. The caller supplies a NumPy
Generator. There are four chains by default; `draws` excludes discarded `warmup`.
Defaults follow the live normal-model form: prior mean zero and all precision,
Gamma shape and Gamma rate inputs 0.0001. Positive finite hyperparameters are
required. Invalid input is rejected before drawing; a numerical failure after
sampling starts can consume RNG state. Unrepresentable pooled statistics or
sampler states raise errors rather than producing substitute values.

Raw `group_mean` and `observation_precision` arrays have axes (chain, draw, group).
`global_mean` and `between_precision` have axes (chain, draw). Stored arrays are
owned and read-only. Sample statistics and pooled statistics are also returned.

## Descriptive comparisons versus posterior uncertainty

Page 6 of the manual labels its no-borrowing curves with empirical group means
and approximately their sample standard deviations. These are observed-outcome
normal fits, not standard errors of group means. The pooled IID curve likewise
uses pooled observed-data spread. `empirical_density` supplies these descriptive
normal fits explicitly. Their sample variances must be positive; the hierarchy
can still analyze constant data even when an empirical normal density is not
defined. The manual's rounded simulated plot labels are not treated as exact
posterior benchmarks.

Posterior uncertainty for theta_i and mu is obtained from `group_mean` and
`global_mean` draws. Data-only confidence intervals for group means are available
through `fit.sample.confidence_interval()` when group sizes permit. These three
objects—observed-outcome spread, data-only confidence intervals, and hierarchical
posterior distributions—remain distinct in the Python API.

## Diagnostics and validation

The shared `summarize_chains` returns empirical shortest intervals, classical
split R-hat and batch-means Monte Carlo error estimates. See
[its documented limits](hierarchical-binomial.md). Weak precision priors, including
the application defaults, can mix slowly. A successful run does not assert
convergence; inspect group/global means and both precision levels, and retain
draws for stronger diagnostics if needed.

Five focused tests check raw versus summary input, empirical means/variances,
pooled statistics, reproducibility, constant-data handling and validation. A joint
Gaussian posterior assembled independently from its precision matrix verifies
means and covariances in a concentrated-precision-prior limit. Integrating out
unknown observation precision gives a separate one-dimensional quadrature
reference for group means and precision expectations. These checks validate the
sampler in the tested regimes, not universal convergence for all datasets.
