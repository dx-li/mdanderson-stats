# Interval-censored proportional-hazards survival

Interval censoring means that an event occurred between two observation times,
rather than at a known exact time. This model fits a proportional-hazards
regression directly to those observation intervals, estimating a nonparametric
baseline distribution on Turnbull maximal-intersection intervals.

```python
import numpy as np
from mdanderson_stats import fit_interval_survival, predict_interval_survival

index = np.arange(1, 97)
x = np.column_stack((np.cos(index*.71) + index/200,
                     np.sin(index*1.31) - index/250))
probability = ((index*37) % 97 + .5)/97
latent = 3*(-np.log(probability)/np.exp(.45*x[:, 0] - .3*x[:, 1]))**(1/1.4)
lower = np.floor(latent)
upper = lower + 1
exact = (index % 7 == 0) & (latent < 6)
lower[exact] = upper[exact] = latent[exact]
lower[latent >= 6] = 6
upper[latent >= 6] = np.inf

fit = fit_interval_survival(lower, upper, x)
prediction = predict_interval_survival(fit, [0, 1, 2, 4.5, 6, 8], [[0, 0], [1, -.5]])
assert np.all(prediction.survival_lower <= prediction.survival_upper)
```

## Observation and model meanings

Ordinary observations are `(lower, upper]`: the event is known to occur after
the lower endpoint and at or before the upper endpoint. Equal endpoints
represent an exact event. An infinite upper endpoint represents right
censoring; a zero lower endpoint represents left censoring on the nonnegative
time scale. Exact event times must be positive. Covariates are finite numeric
columns supplied by the caller; omit them for a Turnbull baseline-only fit.

The model is `S(t | x) = S0(t)^exp(x beta)`. A positive coefficient increases
the hazard. The fitted baseline corresponds to the covariate centering profile;
there is no separately identifiable regression intercept. Case weights multiply
the log-likelihood, with integer weights equivalent to replicated observations.
Exact observations contribute a probability jump at their time, so this is the
nonparametric interval likelihood, not the ordinary Cox partial likelihood.

The fit reports its likelihood, iteration count and normalized score/constraint
diagnostic. A failed convergence check raises an error. Covariates and weights
are internally rescaled, while coefficients and the reported likelihood retain
the input units. Open interval endpoints are compared directly, without adding
a fixed time offset that would change meaning when time units change.

## What prediction bounds mean

The data identify how much probability belongs to each support interval, but
may not identify where that probability lies inside it. The returned lower and
upper survival surfaces describe this uncertainty in event location. They are
**identification bounds, not confidence intervals**. They coincide where the
fitted survival probability is identified. No interpolation silently assigns
events to an interval midpoint.

Right-censored observations can leave probability beyond the last finite
observation. The bounds preserve that tail uncertainty without extrapolating a
parametric tail or placing a fabricated failure at a finite time. Statistical
uncertainty in fitted coefficients and the baseline is separate; this interface
does not invent a Wald covariance or confidence band for the nonparametric fit.

Fits accept up to 20,000 observations, 100 numeric covariates and 2,000 support
intervals, subject to the work limit. Prediction and contour functions check
their combined allocation budgets before constructing surfaces. Iteration,
support, work and output settings can tighten limits within the documented
public function bounds; they cannot disable the hard resource ceilings.

## Continuous-covariate contours

The contour builder reuses a fitted model and varies one numeric covariate.
Other covariates stay at their reference-population means or at an explicit
profile. It returns both identification surfaces and curves at selected
covariate percentiles. Plotting requires an explicit choice of bound.

```python
from mdanderson_stats import interval_survival_contour, plot_interval_survival_contour_2d

contour = interval_survival_contour(fit, x, 0, times=np.linspace(0, 8, 81))
ax = plot_interval_survival_contour_2d(contour, bound="upper")  # optional plotting extra
```

Source provenance and native numerical evidence are recorded in the
[interval-model audit](../research/interval-survival-audit.md). The reference
is the explicitly interval-censored `icenReg` PH likelihood. The original
SurvivalContour application's advertised `mets` interval2 route has an
unresolved response-contract mismatch; this interface does not claim parity
with that route or reuse its counting-process interpretation.

The broader SurvivalContour entry remains partial. Stratified interval models,
bootstrap uncertainty, interval-censored competing risks, neural model workflows
and other outstanding application features are tracked separately in the catalog.
