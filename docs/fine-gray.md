# Fine–Gray competing-risk regression

`fine_gray` fits the subdistribution-hazard model for one target cause in
right-censored competing-risk data. `fine_gray_predict` returns the fitted
cumulative incidence at covariate profiles. This supplies the statistical model
behind SurvivalContour's Fine–Gray family (catalog entry 166).

```python
from mdanderson_stats import fine_gray, fine_gray_predict

fit = fine_gray(
    time=[1, 2, 2, 3, 4, 5, 6, 7],
    status=[1, 2, 0, 1, 2, 1, 0, 1],
    x=[0.2, 0.7, -0.4, 0.1, 0.9, -0.8, 0.3, 0.6],
)
prediction = fine_gray_predict(fit, [[0.2], [0.8]], times=[0, 1, 2, 4, 7])
assert prediction.cumulative_incidence.shape == (2, 5)
assert (prediction.cumulative_incidence[:, 0] == 0).all()
```

The status codes default to 1 for the target cause and 0 for censoring. All
other event codes are competing causes. `failcode` and `cencode` select other
codes. Provide one observation per subject, nonnegative finite times and an
already encoded numeric design without an intercept. Missing observations are
rejected rather than silently removed. Fitting is unpenalized; a finite,
identified coefficient estimate is required.

## Model and uncertainty

Subjects who experience a competing event remain in the subdistribution risk
set. Their later contribution is weighted by the ratio of estimated censoring
survival at the target-event time to that at their own event time. Censoring
survival uses exact left limits; censors tied with a target event remain in the
risk set. Tied target events share one risk denominator, following `cmprsk`'s
Breslow convention. Only exactly equal times are tied.

Use `censoring_groups` to estimate separate censoring distributions when needed.
These groups share target-cause coefficients and baseline hazard. They are not
strata with separate target baselines. This interface assumes independent
subjects and censoring appropriate to the supplied groups; it does not fit
clustered, delayed-entry, interval-censored or longitudinal covariate models.

`fit.coefficients` contains log subdistribution hazard ratios. Its `covariance`
is the sandwich estimate, including the native `cmprsk` estimated-censoring
correction. `observed_information` is a separate result; its inverse is not a
substitute for this covariance. The fit also retains fitted and null pseudo-
log-likelihood, score and residual contributions at distinct target-event
times. The [source audit](../research/fine-gray-audit.md) records the exact
native censoring-group variance convention.

## Time interactions

Supply `time_covariates` and `time_functions` together. The callback receives
the sorted unique target-event times and returns one column for each time
covariate. At time t, each column is multiplied by its corresponding time
function value. A covariate can enter both `x` and `time_covariates` to model
a fixed effect plus a time interaction. All coefficient positions follow fixed
columns first, then time-interaction columns. Models with only time interactions
use `x=None`.

For example, an `n`-by-1 time-covariate matrix paired with
`time_functions=lambda t: (t / 10)[:, None]` adds its interaction with t/10.
Predictions require the corresponding new time-covariate matrix, but no new
callback. The fit retains the evaluated time functions at its event times.

## Prediction conventions

Prediction rows index profiles; columns index requested times. Hazard
increments occur only at fitted target-event times. Values at an event time
include that increment, values before the first event are zero, and curves
stay flat beyond the last target event. An event at zero produces positive
incidence at zero. Time interactions also follow this step convention;
the callback is not evaluated at arbitrary new prediction times.

Incidence is computed as `-expm1(-H)` from the cumulative subdistribution
hazard. The complementary probability is subdistribution survival for this
cause; it is not survival free of every event. Fitting separate models for
several causes does not impose a joint sum-to-one constraint on their predicted
incidences. Baseline estimates and coefficients must be used together, with
the fit's centering and scaling conventions.

`log_baseline_increments` retains the baseline on a log scale. An unscaled
baseline increment can underflow to zero or overflow to infinity after a large
covariate shift, while the fitted coefficients and centered profile predictions
remain representable. Use the prediction function, which combines the centered
log baseline and covariate effects before exponentiating. A cumulative hazard
above the floating-point range is returned as infinity with incidence one.

Incidence confidence bands are not supplied. Coefficient covariance alone
does not include all uncertainty needed for a valid curve confidence interval.
The original SurvivalContour Fine–Gray 3D helper likewise supplies point
predictions without bands.

## Incidence contours

`fine_gray_contour` fits a fixed numeric design and varies one continuous
column. By default, its 30 grid points span the observed 2.5th through 97.5th
percentiles; other columns stay at their training means. Supply `grid` and a
complete `profile` to replace these choices. Default times are distinct
target-event times plus zero if absent. `times` supplies an alternative grid.

```python
from mdanderson_stats import fine_gray_contour, plot_fine_gray_contour_2d

contour = fine_gray_contour(
    time=[1, 2, 2, 3, 4, 5, 6, 7],
    status=[1, 2, 0, 1, 2, 1, 0, 1],
    x=[0.2, 0.7, -0.4, 0.1, 0.9, -0.8, 0.3, 0.6],
    continuous_column=0,
)
assert contour.cumulative_incidence.shape == (30, 5)
ax = plot_fine_gray_contour_2d(contour)  # requires the optional plot extra
```

`plot_fine_gray_contour_3d(contour)` draws the corresponding surface. Both
plotters return Matplotlib axes for labeling and export, require at least two
prediction times, and label the vertical quantity as cumulative incidence.
Plot rendering interpolates between grid points; use the returned arrays for
exact step predictions. Arrays are read-only. The convenience contour builder
uses fixed effects; time-interaction surfaces can be built with the low-level
fit and prediction functions, explicitly constructing the covariate profiles.

The combined contour output is preflighted against a two-million-cell budget
before fitting. The numerical fitter and predictor apply additional work and
allocation limits. The source audit records the executable comparisons and
resource measurements for this implementation.

Fitting accepts at most 100,000 observations, 100 columns and two million input
design values. The product of observations, distinct target-event times and
columns must not exceed two million. Prediction accepts at most 100,000
profiles, two million design values and two million combined profile-by-event
and profile-by-requested-time cells. These conservative limits bound working
arrays; they do not promise a particular total application memory footprint.

Six native `cmprsk` reference fits check coefficients, the full sandwich
covariance, likelihoods, scores, baseline increments and residuals, along with
1,220 incidence values. The contour wrapper additionally matches 800 native
reference values, including the original author's grid/profile conventions.
The largest incidence difference was below 7e-16. The independent numerical
audit peaked at 118.5 MiB; contour checks, guide examples and both plot renders
peaked at 149.3 MiB. Both reported zero process swaps. These are small reference
workloads, not a benchmark of the supported limits.
