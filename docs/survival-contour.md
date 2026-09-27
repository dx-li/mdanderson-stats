# SurvivalContour Cox surfaces

`survival_cox_contour` fits an ordinary right-censored Cox model and predicts
survival over time and one continuous covariate. This implements the first
model family of [SurvivalContour](https://biostatistics.mdanderson.org/shinyapps/survivalContour/),
catalog entry 166. It returns the surface, pointwise confidence limits and
curves at selected covariate quantiles. Plotting uses the existing optional
`plot` extra.

```python
from mdanderson_stats import survival_cox_contour, plot_survival_contour_2d

result = survival_cox_contour(
    time=[1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9],
    event=[1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    x=[[-1, 0], [.3, 1], [1.2, 0], [-.2, 1], [.8, 0], [-1.3, 1],
       [.4, 1], [1.5, 0], [-.7, 1], [.1, 0], [1.1, 1], [-.4, 0]],
    continuous_column=0,
)
assert result.survival.shape == (30, 9)
assert result.quantile_survival.shape == (5, 9)
ax = plot_survival_contour_2d(result)  # optional plot extra
```

`plot_survival_contour_3d(result)` draws a surface. Its `surface` option selects
`"survival"`, `"lower"` or `"upper"`. Both plotting functions return a Matplotlib
axes object for labels and figure export. Numeric predictions need only the
package's core dependencies. Arrays are read-only. The plot renderers interpolate
between the supplied grid points; use the returned step predictions for exact
survival values at a time.

## Model and prediction contract

Supply nonnegative observation times, binary event indicators (1 for an event,
0 for right censoring), and a numeric observations-by-covariates matrix. The
continuous column is identified by its zero-based index. Encode categorical
features before calling this interface; it does not infer factor contrasts.
The design has no intercept. Constant columns, unidentified effects and
monotone likelihood are rejected.

The default Efron tied-event likelihood matches an ordinary R `coxph` fit.
`ties="breslow"` selects the original SURVAN convention. Censors at an event
time remain in its risk set. Only exactly equal times are tied. Covariates
remain fixed throughout follow-up.

The default continuous grid has 30 points evenly spaced between the observed
2.5th and 97.5th percentiles, using linear empirical-quantile interpolation.
Other numeric columns are fixed at their training means. An explicit profile
and grid can replace those choices. The selected continuous column is replaced
by each grid value, regardless of its value in the adjustment profile.

The default time grid contains every distinct observed time, including times
with only censoring. Predictions are right-continuous steps: a value at an
event time includes its hazard increment, survival before the first event is
one, and the curve stays flat after the last event. A caller-supplied time grid
uses the same convention. Surface rows index covariate values and columns
index times.

The additional curve summaries use covariate quantiles, not survival-time
quantiles. Their default probabilities are .1, .25, .5, .75 and .9. This quintet
is an explicit Python choice because the available author helper does not
specify the app's exact five probabilities; callers can change it.

## Pointwise confidence limits

Survival is the exponential of minus the fitted cumulative hazard. Efron or
Breslow hazard increments follow the selected fit convention. The variance
includes uncertainty in both the baseline hazard and fitted coefficients,
using the corresponding baseline/coefficient covariance term. The default
95% log-survival intervals match R `survfit.coxph` with `conf.type="log"` and
have upper bounds capped at one. They are pointwise intervals, not simultaneous
confidence for the entire curve or surface.

This baseline differs from SURVAN's separate Kalbfleisch–Prentice estimator in
`survan_baseline`. See the [implementation audit](../research/survival-contour-audit.md)
for formulas and pinned source provenance.

## Coverage

This interface covers ordinary, unweighted, right-censored Cox models with
static numeric covariates. Stratified and interval-censored Cox models,
parametric/spline models, Fine–Gray cumulative incidence, forests, neural
models and the full native app workflow remain open. Entry 166 stays partial.

The original author's two- and three-dimensional Cox helper outputs provide
reference surfaces for both tie methods and both mean and explicit adjustment
profiles. Independent R `survfit` calls provide quantile-curve references. The
reference generator and four CSV fixtures are included in the source repository.

Input limits are 2..100,000 observations, 1..100 covariates and two million
design entries. Main grids contain 2..2,000 values; quantile summaries contain
1..20 distinct probabilities strictly between zero and one. A combined
two-million-cell budget covers the five main surfaces, five quantile surfaces,
profile matrices and grid vectors. Oversized output is rejected before fitting.
These are allocation bounds, not guarantees about fitting time for large designs.

Risk sets are accumulated with separately scaled event and censor weights.
Cumulative baseline hazard and variance are accumulated in log space, and
prediction standard errors avoid squared-hazard intermediates. Unrepresentable
final hazards or standard errors raise an error. Four R surface comparisons
agree within `8e-16` for probabilities and confidence limits. Scaling time and
the covariates by factors up to `1e100` leaves the predictions unchanged at the
checked tolerance. Additional checks cover a 2,000-log-unit risk separation,
time-step boundaries and allocation preflight. The bounded numerical audit
peaked at 115.4 MiB; rendering both example plots peaked at 161.5 MiB, with no
process swaps reported. No full-suite or large simulation run was needed for
this checkpoint.
