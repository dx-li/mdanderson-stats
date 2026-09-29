# Stratified interval-censored proportional hazards

`fit_stratified_interval_survival` estimates one shared covariate effect and a
separate baseline event distribution for each stratum. It maximizes the sum of
the genuine interval-censored likelihoods. It is useful when groups have
different baseline risks while the covariate's relative effect is shared.

```python
import numpy as np
from mdanderson_stats import (
    fit_stratified_interval_survival,
    predict_stratified_interval_survival,
)

# Frequency-weighted events by time one and event-free observations at time one.
lower = [0, 1, 0, 1, 0, 1, 0, 1]
upper = [1, np.inf, 1, np.inf, 1, np.inf, 1, np.inf]
x = np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, None]
labels = ["A"] * 4 + ["B"] * 4
fit = fit_stratified_interval_survival(
    lower, upper, x, strata=labels, weights=[2, 8, 5, 5, 4, 6, 7, 3]
)
assert abs(fit.coefficients[0] - 0.9624693261) < 1e-6
prediction = predict_stratified_interval_survival(fit, "A", [1], [[0], [1]])
np.testing.assert_allclose(prediction.survival_lower[:, 0], [0.7779801570, 0.5182494399], atol=1e-7)
```

Intervals have the [ordinary interval model's](interval-survival.md) meanings:
`(lower, upper]`, positive exact times at equal endpoints, left censoring at
lower zero and right censoring at infinite upper endpoints. Positive weights
multiply the likelihood; integer weights are frequency weights. Omit
covariates to estimate independent baseline distributions.

Stratum labels are strings or integers, retained in first-seen order. Each
stratum needs at least one finite upper endpoint; entirely right-censored
strata are currently rejected. Coefficients require within-stratum covariate
variation. A stratum indicator cannot also be estimated as a regression
coefficient because its baseline already absorbs that effect.

## Prediction and contours

Each baseline is centered at its own stratum's covariate means. The shared
coefficient vector remains in the original covariate units. Internal scaling
uses pooled within-stratum variation to avoid confusing arbitrary group
offsets with a covariate effect. The fit retains each baseline's support,
probability masses, likelihood and constraint diagnostic.

`fit.for_stratum(label)` returns an immutable `IntervalSurvivalFit` view with
the shared coefficients and matching group baseline and centering. It works
with existing prediction and contour functions:

```python
from mdanderson_stats import interval_survival_contour

contour = interval_survival_contour(
    fit.for_stratum("A"), x[:4], 0, grid=[0, 0.5, 1], times=[0, 0.5, 1, 2]
)
assert np.all(contour.survival_lower <= contour.survival_upper)
```

The contour's reference population is explicit here; use the intended group's
covariates when calculating adjustment means and covariate percentiles. The
existing optional 2D/3D plotting functions accept this contour directly.

Bounds describe unidentified event locations within support intervals. They
are **not confidence intervals**. [Shared-coefficient bootstrap uncertainty](interval-survival-stratified-bootstrap.md)
is available with an explicit within-stratum or pooled resampling policy.
Sampling confidence surfaces remain separate work. This implementation supplies the
advertised statistical family without claiming parity with the original
app's unresolved `mets` interval-response interpretation.

## Validation and limits

Independent base-R complementary-log-log regression provides an exact
current-status special case: shared coefficient, likelihood, separate
baselines and ten predictions. Maximum prediction difference is `3.3e-8`.
A mixed exact/interval/right-censored example also matches a separately
computed survival-difference likelihood; its finite-difference shared score
is below `2.9e-9` per observation. Unit, weight and stratum-offset checks are
recorded in the [audit](../research/interval-stratified-audit.md).

The fitter checks the shared regression score and each constrained baseline,
and raises on failed convergence. Limits are 20,000 rows, 100 covariates,
two million design cells, 2,000 support intervals summed across strata,
20 million estimated likelihood work units and 5,000 iterations. A prediction
call retains the ordinary predictor's combined two-million-cell ceiling.
The one-stratum case uses the existing ordinary fitter.
