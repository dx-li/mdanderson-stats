# Missing data in random survival forests

`fit_random_survival_forest` rejects missing values by default. Set
`na_action="omit"` for complete-case fitting or `na_action="impute"` for the
single-pass tree-local imputation workflow. The chosen policy and a fingerprint
of the original input are recorded in the fit. OOB results expose
`row_indices`, mapping each OOB row back to the caller's input. The fit records
`requested_trees` and the effective `n_trees`; bootstrap trees with no observed
response donors are skipped, so these counts can differ. For a fit where
imputation was performed, OOB concordance is unavailable:
`concordance_available` is false, `concordance_error` is NaN, and
`comparable_pairs` is zero. Complete-data calls retain their existing behavior
and random-number path.

```python
import numpy as np
from mdanderson_stats import fit_random_survival_forest, predict_random_survival_forest

n = 16
time = np.arange(1.0, n + 1.0)
event = (np.arange(n) % 3 != 0).astype(int)
x = np.column_stack((np.linspace(-1, 1, n), np.cos(np.arange(n))))
x[4, 0] = np.nan

imputed = fit_random_survival_forest(
    time, event, x, na_action="impute", n_trees=8, nodesize=1,
    compute_oob=True, random_state=12,
)
prediction = predict_random_survival_forest(
    imputed, [1, 5, 10], [[np.nan, 0.0]], na_action="impute", random_state=13,
)
assert prediction.survival.shape == (1, 3)
assert (np.diff(prediction.survival, axis=1) <= 0).all()

omitted = fit_random_survival_forest(
    time, event, x, na_action="omit", n_trees=8, nodesize=1,
    compute_oob=True, random_state=12,
)
assert 4 not in omitted.training_row_indices
```

## Omission

Omission retains rows with observed time, event status, and every supplied
covariate. `fit.training_row_indices` gives those zero-based input positions in
their original order. OOB curves and contributor counts align with the
retained rows; use the OOB `row_indices` field to map them back. A fit with no
retained event is rejected. Entirely missing time or event inputs are
unrecoverable. Unlike RF-SRC's preprocessing, which drops an entirely missing
predictor column, this interface rejects such a column because it does not
silently change the fitted feature set. The fit also rejects a dataset from
which omission removes every row.

The Brier-score and VIMP adapters accept omitted fits by verifying the full
original-input fingerprint, then applying the fit's row map before their
existing complete-data calculations. Brier rows expose `row_indices` for the
same alignment. This prevents a shifted or modified input from being paired
silently with OOB membership. Contours accept a complete reference matrix and
complete prediction profiles even when the training fit used omission.
Missing contour-reference values have no implicit fill rule and are rejected.

## Single-pass imputation

Imputation is per tree and uses observed in-bag donors; OOB and new prediction
rows are recipients, never donors. Original missing-value masks are retained.
At the first split pass, each candidate feature's cutpoints and split score use
rows with both an observed response and an observed value for that candidate
feature. Imputed predictor values route all rows after the split is selected.
Originally missing responses do not contribute to first-pass split scoring,
but their completed donor values contribute to terminal survival curves.

Donor pools are local to the current node and are restricted to observed values
among that tree's in-bag rows. A missing predictor with no donor in a child
keeps its ancestor-imputed value while routing. At terminal nodes, a missing
response uses the current imputed in-bag values when available. A wholly
missing response variable is unidentifiable and is rejected; a tree bootstrap
with no observed response donors is skipped. Numeric event times use the donor
mean snapped to the forest's master time grid; binary or categorical values
use the modal donor value, with the source tie rule.

This is one imputation pass. It does not implement `nimpute > 1`, which in
RF-SRC aggregates an initial OOB imputation and refits the forest. It also does
not claim native R/C random-stream parity. Missing-aware Brier and VIMP
calculations are not defined by this adapter batch and explicitly reject fits
where imputation was performed. Ordinary prediction accepts complete profiles
from an imputed fit; a profile containing missing values requires the
prediction imputation path and an explicit `random_state`.

The implementation follows the inspected randomForestSRC missing-value
workflow independently. See the committed [missing-data reference audit](../research/random-survival-missing-reference-audit.md)
for source provenance, source-version details, and remaining semantic limits.
