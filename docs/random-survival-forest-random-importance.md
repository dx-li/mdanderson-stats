# Random-routing survival-forest importance

`random_split_random_survival_forest_importance` estimates how much out-of-bag
concordance error changes when tree traversal randomizes splits on one feature.
At the default threshold, a case goes left with probability equal to the left
child's share of the node's fitted sample. Bootstrap duplicates count toward
that share. Splits on other features retain their ordinary routing.

```python
import numpy as np
from mdanderson_stats import (
    fit_random_survival_forest,
    random_split_random_survival_forest_importance,
)

time = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8.0])
event = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
x = np.array([0.2, -1, 0.5, 2, -0.8, 1.5, 0, 1, 3, -0.5, 2.5, 4])
fit = fit_random_survival_forest(
    time, event, x, n_trees=12, nodesize=1, mtry=1, nsplit=0,
    replace=True, ntime=0, random_state=411, compute_oob=True,
)
importance = random_split_random_survival_forest_importance(
    fit, time, event, x, block_size=5, random_state=1772,
)
assert importance.block_count == 2
assert importance.ignored_tree_indices.tolist() == [10, 11]
assert importance.importance.shape == (1,)
assert importance.valid_block_count[0] > 0
print(importance.importance, importance.valid_block_count)
```

This small example demonstrates the interface. It does not establish precise
predictive performance or stable feature rankings.

## Reading the result

Within each complete tree block, baseline and perturbed predictions use only
trees that omitted the observation during fitting. Each prediction averages
the contributing leaf hazards, summed over the fitted time grid. Importance
is the mean perturbed-minus-baseline concordance error over blocks with both
errors defined. Positive values mean prediction deteriorated after perturbation;
negative values are retained. Inspect `valid_block_count`: a feature with no
usable blocks has NaN importance. Incomplete tail trees are reported and omitted.

`block_size=None` uses the entire forest as one block. Supply `block_size=10`
for the pinned native block convention when at least ten trees are available.
Block sizes define different estimators. The result retains baseline errors,
perturbed errors and individual block differences; it supplies no importance
confidence interval.

## Routing, replay and scope

The fit must have `compute_oob=True`, which retains per-node bootstrap sample
counts and packed sampling membership. Pass the original training inputs in
their original row order; the fitted fingerprint verifies them. Explicit
categorical columns use the same stored encoding as prediction. Optional
`feature_indices` selects distinct zero-based columns; grouped-feature
perturbation is not implemented.

The default `vimp_threshold=1` follows the pinned randomForestSRC source. For
a threshold q between zero and one, the source consumes one uniform alpha at
each matching node. If alpha is at most q, the case goes left exactly when
`alpha*q <= L/N`, where N and L include bootstrap multiplicity. Otherwise it
takes the ordinary branch. Both decisions use the same draw. For example,
q=0.4 and L/N=0.5 force all affected draws left; this is not an independent
weighted coin after selecting cases to perturb. At q=0, Python consumes the
draw and preserves the ordinary branch, including the exact alpha=0 endpoint.

Supply exactly one of `random_state` or `rng`. Draw order is block, feature,
tree, ascending out-of-bag row, then tree traversal. This makes Python results
replayable but does not reproduce the native tree-specific random streams.
Work and output bounds are checked before drawing; fitting and importance
run sequentially. See the [source audit](../research/random-survival-forest-audit.md)
and [other importance methods](random-survival-oob.md) for comparison.
