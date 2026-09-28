# Random survival forests

A survival forest fits many log-rank decision trees to right-censored data.
Each leaf estimates a Kaplan–Meier survival curve and a Nelson–Aalen cumulative
hazard curve. Predictions average the leaf curves over trees, allowing nonlinear
covariate effects without specifying a parametric survival distribution.

```python
import numpy as np
from mdanderson_stats import fit_random_survival_forest, predict_random_survival_forest

rng = np.random.default_rng(7)
x = rng.normal(size=(160, 2))
latent = np.exp(1 + 0.6 * (x[:, 0] > 0) - 0.3 * x[:, 1]) * rng.weibull(1.5, 160)
time = np.minimum(latent, 5.0)
event = (latent <= 5.0).astype(int)

# A small reproducible example; the native/default setting is 500 trees.
fit = fit_random_survival_forest(time, event, x, n_trees=32, random_state=17)
prediction = predict_random_survival_forest(fit, [0, 1, 2, 5, 8], [[-1, 0], [1, 0]])
assert prediction.survival.shape == (2, 5)
assert (np.diff(prediction.survival, axis=1) <= 0).all()
```

Inputs are aligned nonnegative observation times, binary `event` indicators
and finite numeric covariates. Event 1 denotes a failure and event 0 right
censoring. All training rows censored is unidentifiable and rejected; an
individual sampled leaf with no events has survival one and cumulative hazard
zero. Missing values and categorical encodings are not inferred.

## Fitting and reproducibility

The default forest has 500 trees. At each eligible node, `mtry` defaults to
`ceil(sqrt(number_of_covariates))`. Up to `nsplit=10` observed-value cutpoints
per selected covariate are sampled without replacement; `nsplit=0` evaluates
all distinct boundaries. The left child contains values less than or equal to
the cutpoint. Cutpoints are observed values, not midpoints.

By default each tree samples `round(.632*n)` rows without replacement.
`replace=True` switches to bootstrap sampling with a default sample size of
`n`. `sample_fraction` explicitly overrides the default fraction. Duplicate
bootstrap rows contribute their multiplicities to event and risk counts.
`nodesize=15` means a parent needs at least 30 sampled rows before splitting;
children can be smaller than 15, matching the inspected native log-rank path.

A fixed `random_state` makes Python runs reproducible. Python's random-number
stream differs from randomForestSRC's, so identical seeds across languages do
not promise identical forests. Numerical evidence and the exact source pin are
recorded in the [source audit](../research/random-survival-forest-audit.md).

## Prediction meanings

Prediction rows index covariate profiles and columns index times. Omitting
`profiles` uses the training covariate means. Omitting `times` uses the fitted
event grid: `ntime=150` retains at most 150 approximately evenly spaced event
order statistics, while `ntime=0` retains all event times. Requested times use
right-continuous leaf curves, including all within-leaf event times. After a
leaf's last event its curve stays constant; the forest does not extrapolate a
parametric tail. An event at time zero contributes its observed jump there.

`survival` is the mean of the leaf Kaplan–Meier curves. `log_survival` is the
logarithm of that mean and can be negative infinity at survival zero.
`cumulative_hazard` is the mean Nelson–Aalen curve. It is generally **not**
`-log(survival)`. These outputs have no pointwise confidence bands; between-tree
variation is not automatically a valid confidence interval.

## Contours from a fitted forest

The contour builder reuses the fitted forest. Its reference matrix supplies
the covariate means and percentiles; columns must match those used for fitting.
The default grid has 30 values from the 2.5th through 97.5th percentile of the
selected column. Other columns stay at their means or at an explicit `profile`.
The result includes curves at selected covariate percentiles as well as the
main surface.

```python
from mdanderson_stats import random_survival_forest_contour, plot_survival_contour_2d

contour = random_survival_forest_contour(fit, x, 0)
ax = plot_survival_contour_2d(contour)  # optional plotting extra
```

Trees grow sequentially. Packed tree data and per-leaf event-step curves avoid
a tree-by-profile-by-time prediction array. Explicit work, stored-event and
output limits reject oversized requests. These bounds limit individual
operations; they do not guarantee total application memory use.

This interface covers ordinary right-censored survival with continuous numeric
predictors. Competing risks, categorical splitting, missing-value imputation,
out-of-bag diagnostics, variable importance and other split rules remain
separate work. It does not claim to reproduce the entire randomForestSRC
package or native random stream. Catalog entry 166 remains partial.
