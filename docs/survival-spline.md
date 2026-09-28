# Spline survival regression

Royston–Parmar models use a natural cubic spline of log time to describe a
survival curve. They extend the Weibull, log-logistic and log-normal families
with flexible baseline shape. `fit_survival_spline` fits all three links and
`predict_survival_spline` returns survival curves and joint-parameter uncertainty.

```python
import numpy as np
from mdanderson_stats import fit_survival_spline, predict_survival_spline

rng = np.random.default_rng(7)
x = rng.normal(size=(100, 2))
latent = np.exp(.8 + .35*x[:, 0] - .25*x[:, 1]) * rng.gamma(2.5, size=100)**.7
time = np.minimum(latent, 6.0)
event = (latent <= 6.0).astype(int)

fit = fit_survival_spline(time, event, x, scale="hazard", k=4)
prediction = predict_survival_spline(fit, [0, 1, 3, 6, 10], [[0, 0], [1, 0]])
assert prediction.survival.shape == (2, 5)
assert (prediction.survival[:, 0] == 1).all()
```

The example uses synthetic data. Observed exact events require positive time
and `event=1`; right-censored observations use nonnegative time and `event=0`.
Covariates are numeric columns without a separate intercept. Omitting covariates
fits a baseline curve alone.

## Model and parameter meanings

Write `z=log(t)` and `eta=B(z) @ gamma + X @ beta`, where the basis has an
intercept, a linear term and one cubic term per internal knot. All three scales
require `d eta / d z > 0` over the full real line.

| Scale | Definition | Survival | Meaning of a covariate coefficient |
| --- | --- | --- | --- |
| `"hazard"` | `eta = log(H)` | `exp(-exp(eta))` | Log hazard ratio |
| `"odds"` | `eta = log((1-S)/S)` | `1/(1+exp(eta))` | Log cumulative failure-odds ratio |
| `"normal"` | `eta = -Phi^-1(S)` | `Phi(-eta)` | Shift in the negative normal survival quantile |

Implementations evaluate these expressions in stable log-domain forms.
Covariate coefficients act on the selected link; they are not the AFT log-time
slopes returned by the other parametric fitters.

With `k=0`, the scales reduce to Weibull, log-logistic and log-normal respectively.
For this case only, the equivalent AFT location is
`mu=-(gamma[0]+X @ beta)/gamma[1]`, and its residual scale is `1/gamma[1]`.

## Knots and valid survival curves

By default, four internal knots are chosen at equally spaced empirical quantiles
of log event times. The boundary knots are the minimum and maximum log event
times. `k` counts internal knots, so `k=4` gives six baseline coefficients.
Alternatively, supply strictly increasing `internal_knots` in **log-time units**.
Repeated quantiles caused by ties require fewer or explicitly chosen knots.

The natural spline has linear tails. The implementation uses these tail
expressions directly, avoiding subtraction of large cubic terms. A proper
survival distribution needs a positive derivative in both tails and every
interior interval. Checking only the observed times is insufficient: the
derivative is quadratic between knots and can reach its minimum inside an
interval. A fitted curve must pass the complete-support check.

The reported parameter order is all baseline spline coefficients followed by
covariate slopes. Joint covariance and observed information use that same
order, including baseline/slope cross terms. Normalized internal coordinates
support fitting and prediction when time or covariate units change.

## Predictions and contours

Prediction rows index profiles and columns index times. Results contain
survival, log survival, cumulative hazard, the standard error of log cumulative
hazard and pointwise lower/upper confidence limits. The limits use the full
joint covariance in a deterministic delta method. These limits are not the
native flexsurv simulated-parameter bounds. At time zero, survival and both
limits are exactly one.

Use `predict_parametric_survival_mc` for the native simulated-parameter
uncertainty method, with shared joint draws across profiles and times:

```python
from mdanderson_stats import predict_parametric_survival_mc

intervals = predict_parametric_survival_mc(
    fit, [0, 1, 3, 6, 10], [[0, 0], [1, 0]], draws=500, rng=166,
)
assert intervals.survival.shape == (2, 5)
assert np.allclose(intervals.survival, prediction.survival)
print(np.count_nonzero(intervals.spline_minimum_slope < 0))
```

Unrestricted Gaussian draws can have negative spline slopes; the native
Monte Carlo summary includes those numerical curves without treating them as
proper fitted survival distributions. `spline_minimum_slope` exposes this
condition for every draw. See [simulation conventions and diagnostics](survival-uncertainty.md).

The common contour workflow accepts `spline_hazard`, `spline_odds` and
`spline_normal`. Other covariates stay at their training means unless a complete
profile is supplied. It retains the common time/grid conventions, percentile
curves, allocation checks and plotting functions. Its default interval method
is `"delta"`; set `interval_method="monte_carlo"` for shared simulated
limits and retained per-draw slope diagnostics. Main and percentile-profile
curves use the same draws.

```python
from mdanderson_stats import parametric_survival_contour, plot_survival_contour_2d

contour = parametric_survival_contour(
    time, event, x, 0, distribution="spline_hazard", spline_k=4,
)
ax = plot_survival_contour_2d(contour)  # optional plotting extra
```

## Current scope

This implementation targets unweighted exact/right-censored observations,
time-constant covariate effects on the intercept of the selected link, and the
native `rp` basis. Interval/left censoring, delayed entry, time-varying effects,
covariates on other spline coefficients and alternate spline bases remain
separate work. Native sources,
reference fixtures and validation status are recorded in the
[spline audit](../research/survival-spline-audit.md). Entry 166 remains partial.
