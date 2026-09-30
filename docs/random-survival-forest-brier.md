# Global Brier-gradient survival splits

`fit_random_survival_forest` offers the optional `split_rule="bs.gradient"`
criterion. Existing fits keep the default `split_rule="logrank"` and its
random stream.

```python
import numpy as np
from mdanderson_stats import fit_random_survival_forest

x = np.arange(8.0)[:, None]
time = np.arange(1.0, 9.0)
event = np.array([1, 0, 1, 0, 1, 0, 1, 0])
fit = fit_random_survival_forest(
    time,
    event,
    x,
    split_rule="bs.gradient",
    prob=0.9,
    nodesize=1,
    n_trees=32,
    random_state=17,
)
assert fit.split_probability == 0.9
```

`prob` is RF-SRC's failure-quantile probability, not a time or an integrated
prediction horizon. It must lie strictly between zero and one; omission uses
the documented native default `0.9`. For a node, let `t_1 < ... < t_m` be its
distinct event times, and `S_j` its parent Kaplan–Meier survival after events
at `t_j`. The C helper scans while `S_j > 1 - prob`, then selects the
immediately preceding event-grid index. With no crossing it selects the last
event point. If the first point already crosses (including equality), the
selected index is zero; RF-SRC skips that QE point and assigns score zero to
each legal split candidate. A scalar probability therefore selects one point,
not the whole history up to a horizon.

At selected event time `t_k`, rows with observed time greater than `t_k` have
`y_i=1`; all other rows have `y_i=0`. The source computes reverse-Kaplan–Meier
censor survival `G` using censor times strictly before the evaluation event
time. Survivors receive weight `1/G(t_k-)`. Every observed failure at or before
`t_k` receives the same weight `1/G(t_{k-1}-)`; for the first event, this is
`1/G(t_1-)`, where `G(t_1-)` includes censorings strictly before the first
event. Censorings at or before `t_k` receive zero weight. These shared
failure weights are the pinned C helper's convention and differ from the
usual subject-specific `1/G(T_i-)` IPCW construction.

The parent weighted fraction is `fhat=sum(w_i*y_i)/sum(w_i)`, with row gradient
`gamma_i=-2*w_i*(y_i-fhat)`. For a left daughter of size `n_L` among `n`
nonmissing node rows, RF-SRC maximizes

```text
(n_L/n) * mean(gamma_left)^2 + ((n-n_L)/n) * mean(gamma_right)^2.
```

The same ordered numeric cuts, categorical subsets, bootstrap multiplicities,
and candidate tie policy as the other forest split rules are used. If required
inverse censor-survival weights cannot be represented, that candidate has no
usable score; if no candidate remains, the node is terminal. No infinite score
or artificial weight floor is introduced. The resolved probability is
available as `fit.split_probability` for this rule and is `None` for the other
rules.

This reproduces the inspected RF-SRC 3.2.2 scalar helper conventions, not
R's random stream or a bit-for-bit native forest. In particular, the manual
describes a 90th-percentile event horizon, while the C helper maps `prob` to a
single preceding event-grid point by parent Kaplan–Meier thresholding.
