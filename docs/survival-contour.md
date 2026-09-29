# SurvivalContour Cox surfaces

`survival_cox_contour` fits an ordinary right-censored Cox model and predicts
survival over time and one continuous covariate. Its stratified counterpart,
`survival_stratified_cox_contour`, fits common covariate effects with separate
group baselines. These implement the right-censored Cox families of
[SurvivalContour](https://biostatistics.mdanderson.org/shinyapps/survivalContour/),
catalog entry 166. It returns the surface, pointwise confidence limits and
curves at selected covariate quantiles. Plotting uses the existing optional
`plot` extra.

```python
from mdanderson_stats import survival_cox_contour, plot_survival_contour_2d

result = survival_cox_contour(
    time=[1, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9],
    event=[1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    x=[
        [-1, 0],
        [0.3, 1],
        [1.2, 0],
        [-0.2, 1],
        [0.8, 0],
        [-1.3, 1],
        [0.4, 1],
        [1.5, 0],
        [-0.7, 1],
        [0.1, 0],
        [1.1, 1],
        [-0.4, 0],
    ],
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

## Stratified Cox models

Use `survival_stratified_cox_contour` when the baseline hazard differs between
groups while the covariate effects are shared. Supply a parallel `strata`
vector of nonempty strings or integers, with at most 100 strata. Labels retain
first-seen order; missing,
nonfinite, boolean and nested labels are rejected. Categorical covariates in
`x` still require caller encoding, but stratum labels do not.

```python
from mdanderson_stats import survival_stratified_cox_contour, plot_survival_contour_2d

# Small illustration: opposite covariate patterns with different follow-up times.
grouped = survival_stratified_cox_contour(
    time=[1, 2, 2, 4],
    event=[1, 0, 1, 0],
    x=[0, 1, 1, 0],
    continuous_column=0,
    strata=["early", "early", "late", "late"],
)
assert grouped.stratum_labels == ("early", "late")
early = grouped.for_stratum("early")
late = grouped.for_stratum("late")
assert early.fit is late.fit
assert list(early.times) == [0, 1, 2]
assert list(late.times) == [0, 2, 4]
ax = plot_survival_contour_2d(late)
```

`fit` contains the shared coefficients, covariance and inference. `contours`
contains one ordinary `SurvivalCoxContour` per label, so both existing plotters
work on each group. The empirical covariate grid, quantile probabilities and
mean/explicit adjustment profile use the whole training sample. Only the risk
sets, baseline estimates and default observed-time grids differ by stratum.
A supplied `times` vector instead evaluates every group on the same times.

Default group timelines include zero if absent. If an event occurs at zero,
the value at zero is post-event, following the right-continuous prediction
contract. A group with no events has estimated survival and pointwise bounds
one; it does not provide evidence that its true event risk is zero. The other
groups must identify the shared coefficients. A covariate constant within
each group cannot be estimated from between-group differences alone.

The original author helper was checked unchanged and repeats the modal group's
predictions across groups with the standard `strata(group)` formula. Its 3D
helper can also return mismatched time and surface dimensions. Python uses the
correctly specified separate group predictions, verified directly against R
`survfit`. The [stratified audit](../research/survival-stratified-audit.md)
records the source versions and executable discrepancy evidence.

Both tie methods, mean and explicit profiles, default and common time grids,
and five covariate-quantile curves are checked against direct R references.
The 780 surface rows and 780 quantile rows include a group without events and
an event at time zero. Probability and confidence-limit differences are below
`7.2e-11`. Additional checks cover joint separation, within-group offsets of
`1e8`, unit and row-order invariance, mixed string/integer labels, and the
combined memory budget. The root numerical check peaked at 118.2 MiB; the
three-group plot peaked at 157.3 MiB, with zero reported process swaps.

## Coverage

These interfaces cover ordinary and stratified, unweighted, right-censored Cox
models with static numeric covariates. [Fine–Gray regression and incidence
contours](fine-gray.md) provide the right-censored competing-risk family.
[Parametric AFT models](parametric-survival.md) add Weibull, log-normal and
log-logistic fitting, prediction and contours.
[Generalized-gamma models](generalized-gamma.md) add the Prentice and Stacy
parameterizations to that workflow. [Spline models](survival-spline.md) add
Royston–Parmar hazard, odds and normal links with joint covariance and contours.
[Interval-censored PH](interval-survival.md), [interval-censored competing-risk
models](interval-competing-risk.md), [numeric survival forests](random-survival-forest.md)
and [simulated parametric intervals](survival-uncertainty.md) provide further
implemented families with their documented numerical and native-parity limits.
[Stratified interval-PH](interval-survival-stratified.md) adds shared regression
effects with group-specific interval baselines.
[Ordinary interval-PH coefficient bootstrapping](interval-survival-bootstrap.md)
adds covariance and standard errors with visible failed replicates.
[Cluster interval-PH bootstrapping](interval-survival-cluster-bootstrap.md)
resamples whole subject groups, and [interval competing-risk bootstrapping](interval-competing-risk-bootstrap.md)
provides coefficient uncertainty for the two-cause model.
[Neural survival models](survival-neural.md) provide DeepSurv, CoxTime,
DeepHitSingle, LogisticHazard and PCHazard fitting, prediction and contours
through a bounded NumPy network with explicit training choices.
[Stratified interval-PH bootstrapping](interval-survival-stratified-bootstrap.md)
adds shared-coefficient covariance with explicit resampling policies and failed-fit records.
Remaining forest features and the full native app
workflow remain open. Entry 166 stays partial.

The original author's two- and three-dimensional Cox helper outputs provide
reference surfaces for both tie methods and both mean and explicit adjustment
profiles. Independent R `survfit` calls provide quantile-curve references. The
reference generator and four CSV fixtures are included in the source repository.

Input limits are 2..100,000 observations, 1..100 covariates and two million
design entries. Main grids contain 2..2,000 values; quantile summaries contain
1..20 distinct probabilities strictly between zero and one. A combined
two-million-cell budget covers the five main surfaces, five quantile surfaces,
profile matrices and grid vectors. Oversized output is rejected before fitting.
For stratified predictions this budget applies to all groups combined, not to
each group separately.
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
