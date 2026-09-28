# Simulated pointwise survival confidence intervals

`predict_parametric_survival_mc` uses joint asymptotic-normal parameter draws
to calculate pointwise survival confidence limits. It accepts the existing
Weibull, lognormal and log-logistic `ParametricSurvivalFit` results and both
Prentice and Stacy `GeneralizedGammaFit` results, plus hazard-, odds- and
normal-link `SurvivalSplineFit` results. This supplies the simulated
parameter-uncertainty method used by flexsurv for these model families.

```python
from mdanderson_stats import fit_parametric_survival, predict_parametric_survival_mc

time = [1, 2, 2, 3, 4, 5, 6, 7]
event = [1, 1, 0, 1, 1, 0, 1, 1]
x = [0.2, 0.7, -0.4, 0.1, 0.9, -0.8, 0.3, 0.6]
fit = fit_parametric_survival(time, event, x, distribution="weibull")
intervals = predict_parametric_survival_mc(
    fit,
    [0, 1, 2, 4, 7],
    [[0.2], [0.8]],
    draws=500,
    rng=2026,
)
assert intervals.survival.shape == (2, 5)
assert (intervals.lower[:, 0] == 1).all()
assert (intervals.upper[:, 0] == 1).all()
```

Rows index covariate profiles and columns index prediction times. Point
estimates use the fitted parameters. Lower and upper limits are empirical
quantiles of survival calculated from the parameter draws, with linear
interpolation corresponding to R's default type-7 quantile. Confidence
defaults to .95. All profiles and times reuse the same joint draws, preserving
the dependence induced by the fitted parameters and their cross-covariances.

These limits describe uncertainty in a survival curve. They are not
simultaneous confidence bands or event-time prediction intervals for a person.
They inherit the fitted model's assumptions and the asymptotic-normal
approximation. Increasing `draws` reduces simulation variability; it does not
fix an unsuitable model or insufficient data.

## Reuse draws for another prediction grid

The result retains the parameter draws in the fit's **normalized coordinates**,
with columns ordered as `fit.scaled_parameters`. They are not draws in the
original-unit `fit.coefficients` coordinates. Reuse them with the same fit:

```python
more_times = predict_parametric_survival_mc(
    fit,
    [0, 0.5, 1, 3, 8],
    [[0.2], [0.8]],
    parameter_draws=intervals.parameter_draws,
)
assert more_times.survival.shape == (2, 5)
```

When parameter draws are supplied, their row count determines the number of
draws; `draws` controls generation only. This supports consistent comparisons
across grids without consuming a second random stream. Do not transfer draws
between fits or between the two generalized-gamma parameterizations. Converting
Stacy parameters to Prentice is nonlinear and does not preserve a Gaussian
sampling distribution.

## Contours with simulated limits

The shared contour interface supports the same Monte Carlo method for all
five parametric choices and all three spline links:

```python
from mdanderson_stats import parametric_survival_contour

contour = parametric_survival_contour(
    time,
    event,
    x,
    0,
    distribution="weibull",
    n_grid=8,
    times=[0, 1, 2, 4, 7],
    interval_method="monte_carlo",
    draws=500,
    rng=2026,
)
assert contour.interval_method == "monte_carlo"
assert contour.lower.shape == (8, 5)
assert contour.monte_carlo.parameter_draws.shape[0] == 500
```

The main grid and selected covariate-percentile curves share one set of
parameter draws. `lower`/`upper` and `quantile_lower`/`quantile_upper` use the
chosen interval method. The attached `monte_carlo` result contains the main
grid rows first, followed by the percentile-profile rows, with their simulated
SDs, evaluable-draw counts and spline slope diagnostics. The contour's existing
`se_log_cumulative_hazard` fields remain delta-method quantities; they are
not the simulated survival SDs.

`interval_method="delta"` remains the default. Supplied `parameter_draws`
require Monte Carlo mode and must use this fit's normalized parameter
coordinates. A shared seed or supplied draws makes repeated grids comparable;
the wrapper checks aggregate output and computational limits before fitting.
Both plot helpers accept the result, and the three-dimensional helper's
`surface="lower"` or `surface="upper"` selects the simulated limit surface.

## Spline models and unrestricted coefficient draws

Spline draws include every baseline coefficient and covariate slope, with
their full joint covariance. Knots and the selected link remain fixed. Draws
use the same normalized coordinates as the fitter. Omitting `profiles`
predicts at raw covariates of zero; supply the training mean explicitly when
comparing with flexsurv's default mean profile.

The native method draws spline coefficients from an unrestricted Gaussian
approximation. Some draws can therefore produce a rising curve on part of the
time axis, even though the fitted curve is monotone. Such a draw does not
define a proper survival distribution. The native Monte Carlo summary keeps
its finite numerical predictions; filtering or redrawing would change those
limits. This implementation follows that convention and exposes
`spline_minimum_slope`, one minimum baseline derivative per draw over the
entire normalized log-time axis. Negative values identify rising sampled
curves. A nonfinite diagnostic means the derivative calculation exceeded its
numerical range. The field is `None` for the other model families.

`valid_draws` counts nonmissing numerical evaluations, including nonmonotone
spline draws; it is not a count of proper survival distributions. The native
endpoint overrides remain `S(0)=1` and `S(infinity)=0` even for such draws.
Inspect the slope diagnostics when interpreting intervals from a diffuse
spline covariance. The [spline uncertainty audit](../research/survival-spline-uncertainty-audit.md)
records the source behavior and independent reference curves.

For zero internal knots the fitted spline models reduce to the three AFT
families, but the mapping from spline to AFT coefficients is nonlinear.
Separate Gaussian approximations in those coordinates need not give identical
simulated limits. Supplied draws mapped through the exact parameter conversion
do give the same curves.

## Missing draws and computational limits

The result includes `simulated_sd` and `valid_draws` for every profile/time
pair. Draws outside the evaluator's numerical range produce missing survival
values. The confidence limits omit those values separately at each point;
they do not truncate or redraw the parameter sample. No valid draws means
missing limits. Sample SD uses the ordinary `B-1` denominator and propagates
missing values, following the source's separate SD convention. Inspect the
valid counts before interpreting limits from a diffuse fitted covariance.

Time zero and positive infinity give survival one and zero for evaluable
draws. Infinite tail arguments are handled point by point, so an extreme
prediction time does not discard an otherwise valid draw at other times.
All returned arrays, including retained parameter draws, are read-only.

The implementation bounds the combined result/parameter allocation, permits
2–100,000 draws and at most 20 million draw-by-profile-by-time evaluations,
and processes times in blocks containing at most 200,000 simulated survival
values. Spline work also accounts for basis width and the per-draw slope
diagnostic. Oversized requests fail before allocating the simulation workspace.
No parallel worker processes are created.

## Source and validation

The [source audit](../research/survival-uncertainty-audit.md) records the pinned
flexsurv 2.3.2 sampling, transformation and quantile conventions. Independent
base-R references cover six cases across the five parameterizations, including
both signs of Prentice Q, full parameter cross-covariances and tail probabilities.
Matching supplied draws separates numerical agreement from random-stream
differences; matching native seeds is not claimed.

The existing `predict_parametric_survival`, `predict_generalized_gamma` and
`predict_survival_spline`
functions retain their deterministic delta-method limits. The simulated
limits are a separate choice; neither interval method guarantees finite-sample
coverage for an arbitrary fitted data set.
