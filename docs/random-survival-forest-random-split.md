# Random survival-forest splits

`fit_random_survival_forest` accepts `split_rule="random"` for the RF-SRC
random-splitting rule. It selects one valid split without evaluating a
survival separation score:

```python
import numpy as np
from mdanderson_stats import fit_random_survival_forest

time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
event = np.array([1, 0, 1, 0, 1, 0])
features = np.column_stack((np.arange(6.0), [0, 0, 1, 1, 2, 2]))
fit = fit_random_survival_forest(
    time,
    event,
    features,
    categorical_features=[1],
    split_rule="random",
    nodesize=1,
    mtry=1,
    n_trees=32,
    random_state=17,
)
assert fit.split_rule == "random"
```

At each eligible node, RF-SRC samples up to `mtry` permissible features
without replacement. A feature with fewer than two observed node values
consumes one `mtry` slot and becomes impermissible below that node. Continuous
features receive one uniformly selected cut from their distinct observed
values except the maximum; values at or below the cut go left. Categorical
features receive one random unordered partition of the observed levels. The
partition generator weights each represented subset-size group by its number
of distinct complementary partitions, then samples uniformly within that
group. This gives every unordered partition equal probability.

The first feature with a valid split wins. There is no log-rank or Brier score
comparison between candidate cuts or features, and the `nsplit` argument does
not change this rule: RF-SRC's random branch generates exactly one candidate
for the selected feature. Parents split only when they contain at least
`2 * nodesize` sampled rows, while valid children need only be nonempty.

For this Python API's no-missing, uniform-feature-weight setting, feature
selection follows the RF-SRC fast path: when `mtry > 1` covers every
permissible feature, features are considered in stored order without feature
selection draws. Otherwise features are drawn sequentially without
replacement; `mtry=1` remains a random feature draw, even when only one feature
is available. The NumPy stream is deterministic for a supplied seed but does
not reproduce RF-SRC's R/C random stream.

The [censoring-forest Brier evaluator](random-survival-oob-brier.md) uses this
rule with 50 trees and the source helper's sample-size-dependent `nodesize`.
See the [source audit](../research/random-survival-forest-random-split-audit.md)
for the split contract and its native source locations.
