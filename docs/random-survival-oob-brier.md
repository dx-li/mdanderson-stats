# OOB Brier score and integrated CRPS for survival forests

`random_survival_forest_oob_brier_score` evaluates the fitted forest's
out-of-bag survival curves over the complete training population. It returns
the per-row Brier contributions, their mean at each event-interest time, the
censoring survival weights, and the time-integrated score (CRPS).

```python
import numpy as np
from mdanderson_stats import (
    fit_random_survival_forest,
    random_survival_forest_oob_brier_score,
)

time = np.array([1, 2, 2, 3, 4, 5, 6, 7.0])
event = np.array([1, 0, 1, 1, 0, 1, 0, 1])
x = np.arange(time.size, dtype=float)[:, None]
fit = fit_random_survival_forest(
    time, event, x, n_trees=24, nodesize=1, compute_oob=True, random_state=51
)
result = random_survival_forest_oob_brier_score(fit, time, event, x)
assert result.brier.shape == (time.size, result.time_grid.size)
assert result.score.shape == result.time_grid.shape
print(result.crps, result.crps_standardized)
```

This is a full-training OOB evaluator. Pass the original times, event
indicators, covariates, and row order used to fit the forest. A fingerprint
check rejects different or reordered training data. All training outcomes
contribute to the censoring distribution; only rows with at least one OOB
prediction contribute to score means. A row without OOB contributors remains
NaN in `brier`. `valid_row_count` reports the number of rows with OOB curves.
`score_row_count` reports the number of rows that contribute a defined value
at each time point. For the default KM model, these counts agree. If no rows
contribute at a time, that score and the integrated summaries are NaN.

Let `tau_i` be observed time, `delta_i` be one for an event, and `S_i(t)` be
the OOB survival estimate. At each grid time `t`, a still-at-risk row
(`tau_i > t`) contributes
`(1 - S_i(t))^2 / G(t)`. An observed event by `t` contributes
`S_i(t)^2 / G_i`; a censoring at or before `t` contributes zero. `G` is the
source's censoring-survival estimate: at each censor time `c`, the increment is
`d_c / Y_c`, where `Y_c` counts all training observations with `tau >= c`, and
`G` is `exp(-sum(d_c/Y_c))`. It is projected to the forest's event-interest
grid. The event denominator uses the projected value at the greatest grid
time less than or equal to `tau_i`; ties therefore use the post-censor value
when a censor time projects to that event time. This reproduces the pinned
helper's `sIndex` convention rather than substituting a product-limit estimate
or a left limit at the raw event time.

`score` averages the row contributions at each time using only rows with OOB
curves. `crps` applies the trapezoidal rule on the returned grid as-is: no zero
time is inserted. The standardized value divides by the largest grid time. A
singleton grid has zero area; if that time is zero, the standardized value is
undefined and returned as NaN. A score that cannot be represented raises an
arithmetic error. The implementation bounds output/workspace cells and
arithmetic work before allocating the row-by-time result.

The default `censor_model="km"` path uses the source's global censoring
estimate and preserves the original behavior. Set `censor_model="rfsrc"` to
fit a separate censoring forest from the complete training cohort. That model
uses censoring as its event, all original training predictors, 50 trees,
`nsplit=1`, random splitting, and the source `set.nodesize` rule; it predicts
curves for each original training row. Pass `censor_random_state` to make this
additional fit reproducible. The result then exposes a row-by-time
`censor_survival` matrix and the fitted `censor_forest_fit`; when there are no
censored observations, it skips the fit and returns an all-one matrix.

```python
forest_scores = random_survival_forest_oob_brier_score(
    fit, time, event, x, censor_model="rfsrc", censor_random_state=52
)
assert forest_scores.censor_forest_fit.n_trees == 50
assert forest_scores.censor_survival.shape == forest_scores.brier.shape
print(forest_scores.score_row_count, forest_scores.crps)
```

The source fits with `na.action="na.omit"` by default but separately overlays
stored imputed covariates for prediction. This Python route requires finite
complete inputs and does not guess the native missing-value/imputation
workflow. Zero censor-survival values are not clipped. Literal source weighting
can yield NaN for an inactive `0/0` term; those row/time contributions are
omitted from that time's mean and reflected in `score_row_count`. A positive
weight that produces an infinite contribution raises an arithmetic error;
literal `0 * Inf` results remain NaN and are omitted as in the source. The native
no-censoring `rfsrc` branch has a vector/matrix shape defect; Python returns
the direct `G(t)=1` result in that case.

Arbitrary subsets remain unsupported because cached native subset handling is
ambiguous. Native forest/RNG parity is not claimed. The
[source audit](../research/random-survival-oob-brier-audit.md) records
the pinned equations and validation evidence.
