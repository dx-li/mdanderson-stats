# Hothorn–Lausen survival split scores

The random survival forest fitter now offers an opt-in `split_rule="logrankscore"`
criterion. The default remains `split_rule="logrank"`, so existing fits and
random streams retain their behavior.

```python
import numpy as np
from mdanderson_stats import fit_random_survival_forest

x = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
time = np.array([1, 2, 3, 4, 5, 6], dtype=float)
event = np.array([1, 0, 1, 0, 1, 0])
fit = fit_random_survival_forest(
    time,
    event,
    x,
    split_rule="logrankscore",
    nodesize=1,
    n_trees=32,
    random_state=17,
)
assert fit.split_rule == "logrankscore"
assert any(tree.feature[0] >= 0 for tree in fit.trees)
```

For a node containing `n` sampled rows, let `Gamma(t)` be the number of node
rows whose observed time is at most `t`, and let `d(t)` be the number of events
at `t`. The Hothorn–Lausen score assigned to row `i` is

```text
a_i = event_i - sum_{t <= time_i} d(t) / (n - Gamma(t) + 1).
```

Equal observed times share their maximum rank, as in the documented Hothorn–Lausen
tie convention. For a candidate left daughter of size `n_L`, the split statistic
is the absolute centered score sum divided by its finite-sample standard
deviation:

```text
abs(sum_{i in left}(a_i - mean(a))) /
    sqrt(n_L * (1 - n_L/n) * sample_variance(a)).
```

The fitter maximizes this statistic over the same feature and split candidates
used by its existing log-rank rule. `nsplit`, categorical-subset generation,
and candidate tie ordering therefore keep their existing meanings. Bootstrap
multiplicities appear as repeated node rows when calculating ranks and scores.
A constant score vector carries no split information and receives score zero.

This implements the published Hothorn–Lausen rank transform as described in
the randomForestSRC survival vignette and `coin::logrank_trafo`. Inspection of
RF-SRC 3.2.2 found index/order inconsistencies in its optional `SURV_LRSCR`
branch. The Python criterion uses the documented scores and candidate-row
mapping; it does not claim bug-for-bug numerical parity with that native branch.
The ordinary log-rank path remains the source-pinned default.
