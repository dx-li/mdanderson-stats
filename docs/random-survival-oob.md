# Out-of-bag survival-forest diagnostics

`fit_random_survival_forest(..., compute_oob=True)` adds predictions for each
training observation using only trees that did not sample that observation.
This provides a prediction-error diagnostic without using the observation's
own training trees. Ordinary fitting and prediction remain available with
OOB computation disabled, which is the default.

```python
import numpy as np
from mdanderson_stats import fit_random_survival_forest

time = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8.0])
event = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
x = np.array([0.2, -1, 0.5, 2, -0.8, 1.5, 0, 1, 3, -0.5, 2.5, 4])
fit = fit_random_survival_forest(
    time, event, x,
    n_trees=12, nodesize=1, mtry=1, nsplit=0,
    sample_fraction=0.75, ntime=0, random_state=411,
    compute_oob=True,
)
assert fit.oob is not None
assert fit.oob.survival.shape == (12, 7)
assert fit.oob.comparable_pairs == 44
print(fit.oob.concordance_error)  # 0.31818181818181823
```

The small example illustrates the API; it is not a claim of precise
out-of-sample performance. `oob.contributor_count` reports how many trees
contribute to each row. A row with no contributors has `NaN` curves and
mortality and is excluded from error estimation. It is never filled using
in-bag predictions. An error with no comparable pairs is also `NaN`.

Survival averages the contributing leaf Kaplan–Meier curves. Cumulative hazard
averages their Nelson–Aalen curves separately. `mortality` sums the latter
over the fitted interest-time grid; it is not a time integral or the final
hazard value. Changing `ntime` can therefore change mortality rankings and
concordance error, matching the native implementation. Curves are returned at
the fitted grid, including a survival jump at time zero where present.

`concordance_error` is `1-C`: lower error means better ordering of observed
survival. The native rule includes a pair when the earlier time is an observed
failure. It orders a failure before censoring at tied times. For two tied
failures, equal mortality gets full concordance and different mortality gets
half; ordinary comparable pairs with tied mortality get half. The original
absolute `1e-9` time/mortality tolerances are preserved, including their
different strictness at exact equality. Changing time units can alter which
near-tied times are treated as tied. This convention need not equal the
concordance implementation in another package.

`fit.inbag_membership` stores one packed byte row per tree, using little-endian
bit order; bit one means the training observation was sampled at least once.
Bootstrap multiplicities affect fitting, but membership records presence only.
The arrays are read-only. No extra random draws are consumed by OOB processing,
so enabling it preserves the fitted trees for an identical seed.

`max_oob_cells` and `max_oob_work` bound combined OOB buffers and computation.
Pair comparisons use row-sized temporary vectors instead of an observation-by-
observation matrix. The [source audit](../research/random-survival-forest-audit.md)
records exact agreement with 16 native concordance cases and 264 independently
reconstructed native-kernel curve values. These checks do not establish
native full-forest random-stream equivalence or confidence intervals.

## Permutation importance

`permutation_random_survival_forest_importance` measures the increase in OOB
concordance error after shuffling one feature within each tree's OOB rows.
It requires the original training times, event indicators and covariates in
the original row order. A compact fingerprint verifies these inputs without
retaining a second training-data matrix in the fit.

Continuing the example above:

```python
from mdanderson_stats import permutation_random_survival_forest_importance

importance = permutation_random_survival_forest_importance(
    fit, time, event, x, block_size=5, random_state=1772,
)
assert importance.block_count == 2
assert importance.ignored_tree_indices.tolist() == [10, 11]
assert importance.importance.shape == (1,)
print(importance.importance)
```

Within each complete tree block, predictions are averaged before computing
baseline and perturbed errors. Importance averages their differences across
blocks with defined errors. Positive values indicate worse predictions after
permutation; negative values are retained. Blocks without comparable OOB pairs
are excluded, and `valid_block_count` exposes how many remain. If none remain,
importance is `NaN`. The incomplete final block is excluded explicitly and its
zero-based tree indices are returned.

The Python default `block_size=None` uses all trees as one block. The native
randomForestSRC default for explicit permutation importance is a block size of
10; supply that value to select the same block convention. Its default
`importance=True` instead uses anti-split importance, which this API does not
implement. Smaller blocks and a single whole-forest block are different
estimators. `feature_indices` optionally selects distinct zero-based columns.

Supply either `random_state` or a generator through `rng`. Random permutations
are drawn in block, feature, then tree order, including for constant features.
The result retains baseline errors, perturbed errors, block differences and
valid-block counts. Work limits apply before permutation, and pair comparisons
use row-sized temporary vectors. The independent native-kernel reference checks
whole-forest, three-tree and single-tree blocks, including an ignored tail,
an undefined block and zero importance for a constant feature. Native RNG
equivalence, anti-split/random importance and importance confidence intervals
remain outside this implementation.
