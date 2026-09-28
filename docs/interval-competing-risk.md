# Competing risks with interval-censored event times

This model estimates cumulative incidence for two competing causes when an
event is known to have occurred between visits. Each cause has a monotone cubic
B-spline baseline and its own regression coefficients. The link parameters
`alpha=(0,0)` select proportional subdistribution hazards for both causes;
`alpha=(1,1)` select proportional odds, and mixed values in `[0, 20]` are allowed.

```python
import numpy as np
from mdanderson_stats import fit_interval_competing_risk, predict_interval_competing_risk

index = np.arange(1, 121)
x = np.column_stack((np.cos(index * 0.71) + index / 200, np.sin(index * 1.31) - index / 250))
u1 = ((index * 37) % 127 + 0.5) / 127
u2 = ((index * 53) % 131 + 0.5) / 131
t1 = -np.log(u1) / (0.3 * np.exp(0.4 * x[:, 0] + 0.2 * x[:, 1]))
t2 = -np.log(u2) / (0.25 * np.exp(-0.2 * x[:, 0] + 0.1 * x[:, 1]))
latent = np.minimum(t1, t2)
event = np.where(t1 < t2, 1, 2)
visit_spacing = 0.3 + 0.013 * (index % 11)
lower = np.floor(latent / visit_spacing) * visit_spacing
upper = lower + visit_spacing
censor_time = 2 + 0.023 * (index % 13)
right = upper > censor_time
lower[right] = censor_time[right]
upper[right] = np.inf
event[right] = 0

fit = fit_interval_competing_risk(lower, upper, event, x, alpha=(0, 0), k=0.5, tolerance=1e-9)
times = np.linspace(fit.boundary_knots[0], fit.boundary_knots[1], 51)
prediction = predict_interval_competing_risk(fit, times, [[0, 0], [0.5, -0.25]])
assert np.all(prediction.cif1 + prediction.cif2 <= 1 + 1e-7)
```

## Input and model meanings

`event=0` means right censoring at `lower`; that observation's `upper` value
does not contribute to fitting. Events of cause 1 or 2 use strictly ordered
finite endpoints `(lower, upper]`. A zero lower endpoint represents left
censoring. This smooth interval-likelihood API does not reinterpret equal
endpoints as an exact event.

The model for cause `j` is
`F_j(t | x) = 1 - (1 + alpha_j * exp(eta_j(t,x)))^(-1/alpha_j)`.
Here `eta_j(t,x) = B(t) phi_j + (x - covariate_mean) beta_j`: the returned
baseline coefficients describe the centered reference profile, and the returned
regression coefficients use the original covariate units. At `alpha_j=0`,
the continuous limit is `1 - exp(-exp(eta_j(t,x)))`.
An observed event contributes the increment in its cause's incidence over the
interval. A censored observation contributes `1-F_1(lower)-F_2(lower)`.
The two causes are fitted jointly, with probability constraints; fitting two
independent binary or survival regressions would not impose this model.

Numeric covariates have separate coefficients for each cause. There is no
separate intercept outside the spline baselines. The `k` parameter controls the
number of interior empirical-quantile knots and must be between 0.5 and 1.
Prediction uses the fitted knot sequence and rejects times outside the observed
boundary range rather than extending the model beyond that range.

## Starting-boundary approximation and probability checks

An exactly zero incidence at the starting boundary would require an infinite
log-baseline coefficient. The finite spline model instead exposes
`boundary_cif_tolerance`, defaulting to `1e-7`, and reports the achieved
lower-boundary incidence. Tightening that tolerance can change the fitted curve
inside the first knot span, not just its value at the starting point. It is a
model approximation that should be reported alongside results.

Monotone spline controls ensure that each incidence curve increases over time.
Joint constraints are applied at the maximum supported time. Constraints at
the observed covariate-range corners alone do not guarantee valid predictions
at every interior profile for all nonnegative link parameters. The predictor
checks each requested profile at the fitted maximum time, including when only
earlier predictions were requested, and raises if the two incidences cannot
form a valid joint distribution. It does not silently clip their sum.

Regression covariance follows the source's residualized-score least-squares
method, separating regression scores from the baseline score span. This is
distinct from inverting the full likelihood Hessian. Neither bootstrap
confidence bands nor uncertainty surfaces are implied by the returned curves.

`score_error` reports the ordinary likelihood score in the fitter's scaled
parameter coordinates; it can remain nonzero at an active probability or
monotonicity constraint. `kkt_error` reports the largest remaining score component
per observation after accounting for active constraints with nonnegative
multipliers. Feasibility and this
constrained stationarity check determine whether a fit is accepted. They do not
establish that a nonconvex problem has a unique global optimum.

Fitting accepts up to 20,000 observations and ten numeric covariates, with
additional combined allocation limits. Prediction and contour requests also
have explicit output-size limits. These limits keep a single request bounded;
large grids should be evaluated in smaller batches.

## Continuous-covariate contours

The contour builder reuses the fitted model, varies one numeric covariate and
holds other columns at the reference population's means or an explicit profile.
It returns both incidence surfaces and selects cause 1 or 2 for plotting.
Default covariate values span the 2.5th to 97.5th percentiles; default times are
50 points across the fitted time range.

```python
from mdanderson_stats import (
    interval_competing_risk_contour,
    plot_interval_competing_risk_contour_2d,
)

contour = interval_competing_risk_contour(fit, x, 0, cause=1)
ax = plot_interval_competing_risk_contour_2d(contour)  # optional plotting extra
```

See the [source and numerical audit](../research/interval-competing-risk-audit.md)
for pinned `intccr` sources, the native derivative discrepancies and reference
scope. The original author's contour starts at zero even when that time lies
outside the fitted range; this interface starts at the fitted lower boundary.
Categorical encoding, visit-data conversion and the broader SurvivalContour
application remain separate coverage items.
