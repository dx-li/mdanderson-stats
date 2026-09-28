# Generalized-gamma survival regression

Generalized gamma extends the Weibull and log-normal survival families by
estimating an additional shape parameter. SurvivalContour (catalog entry 166)
offers two parameterizations: stable Prentice and original Stacy. They have
different parameter meanings and ranges.

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

## Implementation status

The native distribution, fitting and prediction references are committed.
The Python fitter and predictor are being implemented and are not yet exposed
by the public package. Usage examples and verified numerical limits will be
added with integration. Generalized-gamma contours, the original simulated
confidence-bound workflow, spline models and the other remaining catalog
methods are not completed by these reference results.
