# Weibull goodness of fit with unknown shape and scale

`weibull_unknown_shape_bayesian_gof` jointly estimates Weibull shape and scale,
then applies the [Johnson posterior diagnostic](bayesian-chi-square.md) to
complete positive event times. It complements the
[fixed-shape exact-posterior workflow](weibull-bayesian-gof.md).

```python
import numpy as np
from mdanderson_stats import weibull_unknown_shape_bayesian_gof

prior_mean = np.log([1.5, 2.0])  # log shape, log scale
fit = weibull_unknown_shape_bayesian_gof(
    [.5, 1, 1.5, 2.2, 3],
    prior_mean=prior_mean, prior_covariance=[[.16, .056], [.056, .49]],
    initial=prior_mean + [[-.2, -.35], [-.2, .35], [.2, -.35], [.2, .35]],
    draws=800, warmup=300, chains=4, bins=3,
    rng=np.random.default_rng(660940),
)
assert fit.parameters.shape == (4, 800, 2)
assert fit.diagnostic.statistic.shape == (3200,)
log_shapes = fit.parameters[..., 0]
log_scales = fit.parameters[..., 1] + fit.log_scale_offset
print(np.exp(log_shapes).mean(), np.exp(log_scales).mean())
print(fit.parameter_summary.split_rhat)
```

## Model and prior

The model follows §4.7 of the
[BCSTTE guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BCSTTE/BCSTTE_UsersGuide.pdf):

```text
f(t | beta, eta) = beta/eta * (t/eta)**(beta-1) * exp(-(t/eta)**beta)
F(t | beta, eta) = 1 - exp(-(t/eta)**beta)
```

The caller supplies a proper bivariate Gaussian prior on `(log(beta),log(eta))`
through `prior_mean` and a symmetric positive-definite `prior_covariance`.
Correlation is supported. These are explicit Python prior coordinates;
the source guide does not establish the native program's prior or fitter.

Serial elliptical slice sampling draws both parameters jointly. Each retained
pair evaluates all observations before the existing diagnostic forms
equal-probability bin counts and its `bins-1` reference statistic. Draws from
this sampler are correlated. The returned `parameter_summary` includes means,
intervals, classical split R-hat and batch-means Monte Carlo errors; these are
diagnostics, not convergence guarantees. Inspect chains and use suitable
dispersed starts and adequate warmup/draw counts for the dataset.

## Numerical representation

`parameters` has axes `(chain, draw, parameter)`. The first coordinate is log
shape; the second is log scale centered by `log_scale_offset`. Adding the
offset recovers absolute log scale. The second coordinate of
`parameter_summary` is centered too. Prior and supplied initial coordinates
are absolute. Returned `log_likelihood` retains the absolute observed-data
log likelihood with the same chain/draw axes.

To change time units by factor `c`, multiply observations by `c` and add
`log(c)` to the prior and initial log-scale coordinates. Covariance is
unchanged. Relative log times and a centered sampling likelihood preserve
this transformation without subtracting large, nearly equal likelihood
terms. Parameter-independent density constants are restored only in the
reported likelihoods.

Likelihood-evaluation, total-work and combined-memory budgets are checked
before sampling and enforced during slice updates. The combined bound includes
CDF/count arrays, retained draws and chain-summary temporaries. Exponentially
large hazards have zero representable likelihood and are rejected; invalid
initial states, unrepresentable shape coordinates, numerical failures and
budget exhaustion raise explicit errors. No approximate fit is returned after
a failed sampler.

Complete continuous observations are required. Censoring, rounding, native
prior defaults, BIC/DIC and native reporting remain separate coverage gaps.
The Johnson reference is asymptotic; posterior-average reference probabilities
are not calibrated p-values for an averaged statistic.

## Validation

Independent base-R quadrature checks two fully bivariate posteriors with
correlated priors and increasing/decreasing hazard examples. Changing the
quadrature order and integration range changes all reference moments/CDFs by
at most `9.47e-10`. Twenty-three posterior means, variances, covariance and
CDF summaries from four chains agree within 2.281 estimated batch-means
Monte Carlo errors; maximum classical split R-hat is 1.00175. Three focused
checks cover density/CDF identities, time-unit invariance and resource bounds.
See the [audit](../research/weibull-unknown-shape-audit.md).
