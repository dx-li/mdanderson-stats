# Generalized-gamma survival regression

Generalized gamma extends the Weibull and log-normal survival families by
estimating an additional shape parameter. SurvivalContour (catalog entry 166)
offers two parameterizations: stable Prentice and original Stacy. They have
different parameter meanings and ranges.

`fit_generalized_gamma` fits either model to exact and right-censored data;
`predict_generalized_gamma` returns survival curves and joint-parameter
uncertainty. The following example uses synthetic data.

```python
import numpy as np
from mdanderson_stats import fit_generalized_gamma, predict_generalized_gamma

rng = np.random.default_rng(7)
x = rng.normal(size=(100, 2))
latent = np.exp(.8 + .35*x[:, 0] - .25*x[:, 1]) * rng.gamma(2.5, size=100)**.7
time = np.minimum(latent, 6.0)
event = (latent <= 6.0).astype(int)

fit = fit_generalized_gamma(time, event, x, parameterization="prentice")
prediction = predict_generalized_gamma(fit, [0, 1, 3, 6, 10], [[0, 0], [1, 0]])
assert prediction.survival.shape == (2, 5)
assert (prediction.survival[:, 0] == 1).all()
```

This example generates synthetic data. For observed data, use one positive
time and `event=1` per exact event, or a nonnegative time and `event=0` per
right-censored observation. Covariates are numeric columns without an intercept;
omit them for an intercept-only model. Use `parameterization="stacy"` to fit
the original model to the same data.

## Model definitions

In the **Prentice** model, the linear predictor is `mu = intercept + X @ slopes`,
sigma is positive, and Q can be positive, negative or zero. For Q different
from zero, if G has a gamma distribution with shape `1/Q**2` and rate one,

```text
log(T) = mu + sigma * log(Q**2 * G) / Q.
```

At Q=0, `log(T)` is exactly normal with mean mu and standard deviation sigma.
At Q=1, T is Weibull with shape `1/sigma` and scale `exp(mu)`. Negative Q
reverses which gamma tail corresponds to survival. Retaining this sign is
essential; replacing Q by its absolute value changes the model.

In the **Stacy** model, the positive parameters are a (scale), b (shape) and k
(gamma shape). The model is `T = a * G**(1/b)`, where G has a gamma distribution
with shape k and rate one. Regression specifies `log(a) = intercept + X @ slopes`.
Its equivalent Prentice parameters are:

```text
Q = 1 / sqrt(k)
sigma = 1 / (b * sqrt(k))
mu = log(a) + log(k) / b.
```

Stacy therefore covers the positive-Q part of the Prentice family. A Prentice
fit with negative Q cannot be converted into a finite positive Stacy fit.
The log-normal limit also requires unbounded original parameters; a numerical
bound on those parameters must not be reported as a finite maximum-likelihood
estimate.

Within either model, covariate slopes are effects on log time when the shape
parameters are held fixed. Exponentiating a slope gives a time ratio. The
intercept and ancillary parameters have different meanings in the two
parameterizations, even when they describe the same survival distribution.

## Likelihood and joint uncertainty

For an exact positive event time t, the likelihood uses the time density f(t).
For right censoring at a nonnegative t, it uses survival S(t). A censor at time
zero contributes log likelihood zero. An exact event at zero is invalid.
Both regression and shape parameters must be estimated together; treating
shape as known omits uncertainty and its correlation with the regression
coefficients.

Joint covariance coordinates are regression coefficients followed by
`log(sigma), Q` for Prentice, or by `log(b), log(k)` for Stacy. Transforming a
positive-Q fit to Stacy coordinates changes the intercept as well as the two
ancillary parameters, so all covariance cross terms must be transformed.

The [source and numerical audit](../research/generalized-gamma-audit.md)
records the exact native kernel and independent R fitting harness. Near-zero
Q requires stable density and tail calculations that retain shape derivatives.
Direct upper/lower gamma tails avoid the loss of small survival probabilities
caused by subtracting a CDF from one.

## Predictions and contours

Prediction rows index covariate profiles and columns index times. Omitting
profiles selects one zero-covariate profile. The result retains survival, log
survival, cumulative hazard, the standard error of log cumulative hazard, and
lower/upper pointwise limits. Limits use a deterministic normal delta method
with the full joint parameter covariance; they do not reproduce flexsurv's
simulated-parameter intervals.

The shared contour workflow uses `distribution="gengamma"` for Prentice and
`distribution="gengamma.orig"` for Stacy. Other columns stay at their training
means unless a complete `profile` is supplied. Its default grid spans the
selected covariate's 2.5th through 97.5th percentiles, and prediction times are
distinct event times plus zero. The same result supports both existing plot
functions and selected covariate-percentile curves.

```python
from mdanderson_stats import parametric_survival_contour, plot_survival_contour_2d

contour = parametric_survival_contour(time, event, x, 0, distribution="gengamma")
ax = plot_survival_contour_2d(contour)  # optional plotting extra
```

## Scope and numerical limits

The fits are unweighted, unpenalized and right-censored, with static numeric
covariates and common shape parameters. The optimizer requires a finite
stationary solution and positive-definite observed information. Separated
location designs and failed convergence raise errors. Two native negative-Q
datasets also reject a finite Stacy optimum as parameters approach its
log-normal boundary; this is not an exhaustive characterization of boundary
behavior for arbitrary data.

Fitting accepts up to 20,000 observations and 16 covariates. Prediction limits
the combined eight retained surface arrays and profile design to two million
cells. Contours include the main grid and percentile curves in their aggregate
allocation check. These are operation limits, not a bound on total application
memory.

The native-reference checks cover 130 distribution rows, six fitted models,
full joint covariance and information, and 120 prediction rows. Additional
checks cover the Q=0 derivatives, tail probabilities, parameterization
equivalence, large changes in units, zero-time censoring and both contour
parameterizations. The audit records tolerances and resource measurements;
small absolute error is not claimed for extremely large log-tail values.

[Simulated pointwise curve limits](survival-uncertainty.md) are available through
`predict_parametric_survival_mc` for both fitted parameterizations. It uses each
fit's own full joint covariance and can reuse parameter draws across grids.
The contour wrapper supports these limits with
`interval_method="monte_carlo"`, sharing draws across the main grid and
selected covariate-percentile profiles. Its default remains `"delta"`.
Interval/left censoring, delayed entry and covariates on ancillary parameters
remain open. Spline models have a separate
[fitter and prediction interface](survival-spline.md). Entry 166 stays partial.
