# Simulated pointwise survival confidence intervals

`predict_parametric_survival_mc` uses joint asymptotic-normal parameter draws
to calculate pointwise survival confidence limits. It accepts the existing
Weibull, lognormal and log-logistic `ParametricSurvivalFit` results and both
Prentice and Stacy `GeneralizedGammaFit` results. This supplies the simulated
parameter-uncertainty method used by flexsurv for these model families.

```python
from mdanderson_stats import fit_parametric_survival, predict_parametric_survival_mc

time = [1, 2, 2, 3, 4, 5, 6, 7]
event = [1, 1, 0, 1, 1, 0, 1, 1]
x = [.2, .7, -.4, .1, .9, -.8, .3, .6]
fit = fit_parametric_survival(time, event, x, distribution="weibull")
intervals = predict_parametric_survival_mc(
    fit, [0, 1, 2, 4, 7], [[.2], [.8]], draws=500, rng=2026,
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
    fit, [0, .5, 1, 3, 8], [[.2], [.8]],
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
values. Oversized requests fail before allocating the simulation workspace.
No parallel worker processes are created.

## Source and validation

The [source audit](../research/survival-uncertainty-audit.md) records the pinned
flexsurv 2.3.2 sampling, transformation and quantile conventions. Independent
base-R references cover six cases across the five parameterizations, including
both signs of Prentice Q, full parameter cross-covariances and tail probabilities.
Matching supplied draws separates numerical agreement from random-stream
differences; matching native seeds is not claimed.

The existing `predict_parametric_survival` and `predict_generalized_gamma`
functions retain their deterministic delta-method limits. The simulated
limits are a separate choice; neither interval method guarantees finite-sample
coverage for an arbitrary fitted data set.
