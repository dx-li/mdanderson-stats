# Parametric accelerated failure-time models

`fit_parametric_survival` fits exact Weibull, log-normal and log-logistic
accelerated failure-time (AFT) models to positive event times and nonnegative
right-censoring times. These are three parametric model families offered by
SurvivalContour (catalog entry 166).

```python
from mdanderson_stats import fit_parametric_survival, predict_parametric_survival

time = [1, 2, 2, 3, 4, 5, 6, 7]
event = [1, 1, 0, 1, 1, 0, 1, 1]
x = [0.2, 0.7, -0.4, 0.1, 0.9, -0.8, 0.3, 0.6]
fit = fit_parametric_survival(time, event, x, distribution="weibull")
prediction = predict_parametric_survival(fit, [0, 1, 2, 4, 7], [[0.2], [0.8]])
assert prediction.survival.shape == (2, 5)
assert (prediction.survival[:, 0] == 1).all()
```

Use `event=1` for an exact event and `event=0` for right censoring. A censor at
zero contributes no likelihood information. An exact event at zero is rejected.
Missing, nonfinite and complex data are rejected. Supply numeric covariates
without an intercept column; the fit adds an intercept. Omit covariates to fit
an intercept-only model. Categorical variables must already be encoded.

## Model and parameters

The model is `log(T) = mu + sigma * Z`, where `mu` is the intercept plus the
numeric covariate effects and `sigma` is a positive residual scale.

| `distribution` | Distribution of T |
| --- | --- |
| `"weibull"` | Weibull shape `1/sigma`, scale `exp(mu)` |
| `"lognormal"` | Log-normal with mean of log(T) `mu` and standard deviation `sigma` |
| `"loglogistic"` | Log-logistic shape `1/sigma`, scale `exp(mu)` |

Coefficients are effects on log time; exponentiating a slope gives a time
ratio. They are not generally log hazard ratios. The fitted `sigma` is not
the Weibull or log-logistic distribution scale shown in the table.

`fit.coefficients` contains the intercept followed by slopes. Both `covariance`
and `information` use **intercept, slopes, log(sigma)** order, including the
cross-covariances between coefficients and scale. The log likelihood includes
the event-time density Jacobian. Centered and scaled fitting coordinates are
retained for stable profile prediction; use the prediction function instead
of combining a large intercept and covariates manually.

## Predictions and uncertainty

Prediction rows index profiles and columns index requested times. Each profile
must supply one value for each fitted covariate. Curves are continuous
parametric predictions at the supplied times, including beyond the last event.
At time zero survival is one and cumulative hazard is zero.
Omitting profiles selects one profile with all covariates zero, or the single
intercept-only curve when no covariates were fitted.

The result contains survival, log survival, cumulative hazard, lower and upper
pointwise bounds, and `se_log_cumulative_hazard`. Bounds use a normal
delta-method approximation on **log cumulative hazard**, propagated through
the full joint covariance and transformed back to survival probabilities.
`confidence` defaults to 0.95. These are pointwise parameter-uncertainty bounds,
not simultaneous bands or prediction intervals for a person's event time.

The original flexsurv workflow obtains curve bounds from simulated asymptotic
normal parameter draws. The deterministic bounds here use a different method;
numerical agreement with native point estimates does not establish interval
parity with that simulation procedure.

## Contours and selected covariate curves

`parametric_survival_contour` fits a model and varies one continuous column,
holding other columns at their means or at a supplied complete `profile`.
Its default 30-point grid spans the empirical 2.5th through 97.5th percentiles.
Default prediction times are distinct event times plus zero; provide `grid`
and `times` to choose other values.
Set `interval_method="monte_carlo"` to use joint-normal simulated confidence
limits, with `draws` and `rng` controlling generation. Main and percentile
profiles share those draws; the attached `monte_carlo` result retains their
diagnostics. The default remains `"delta"`; see the
[simulated-interval guide](survival-uncertainty.md).

```python
from mdanderson_stats import parametric_survival_contour, plot_survival_contour_2d

contour = parametric_survival_contour(time, event, x, continuous_column=0, distribution="lognormal")
ax = plot_survival_contour_2d(contour)  # optional plot extra
```

The same result works with `plot_survival_contour_3d`; select `surface="lower"`
or `surface="upper"` to plot a confidence-limit surface. Returned arrays retain
the actual evaluated values; rendering interpolates between grid points.
Selected curves default to the 10th, 25th, 50th, 75th and 90th covariate
percentiles, an explicit Python convention. `quantile_probabilities` changes
these choices. This concerns covariate percentiles, not event-time quantiles.

## Scope

The current fits are unweighted, unpenalized and right-censored, with one
common residual scale and static numeric covariates. They require an identified
finite maximum-likelihood fit. Interval/left censoring, delayed entry,
covariates on ancillary parameters remain open.
[Simulated pointwise curve limits](survival-uncertainty.md) are
available through `predict_parametric_survival_mc`, which preserves full joint
parameter uncertainty and allows draws to be reused across prediction grids. The separate
[generalized-gamma fitter](generalized-gamma.md) supplies stable Prentice and
original Stacy distributions through the same contour interface. The separate
[spline fitter](survival-spline.md) supplies Royston–Parmar hazard, odds and
normal links, also with predictions and contours.

The [source and numerical audit](../research/parametric-survival-audit.md)
records the native reference versions and checks. Array and work limits bound
individual operations; they do not guarantee total application memory use.
Fitting accepts at most 20,000 rows and 16 covariates. Prediction accepts up to
100,000 profiles and 100,000 requested times, subject to a combined budget of
two million cells counting eight surface arrays and the profile design. Contour
preflight includes both primary and percentile curves in the combined budget.
