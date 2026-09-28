# PLBARPO allocation among active arms

`plbarpo_active_allocation` recomputes posterior competition among the current
active arms and keeps outputs aligned to the complete arm ledger. A closed arm
receives zero best-arm probability and zero allocation, while its enrollment
still contributes to the BARN2N randomization exponent.

```python
import numpy as np
from mdanderson_stats import plbarpo_active_allocation

result = plbarpo_active_allocation(
    successes=[1, 0, 8], failures=[0, 1, 0], assigned=[1, 1, 8],
    active=[True, True, False], prior=[[1, 1], [1, 1], [2, 1]],
    method="barn2n", max_n=20,
)
np.testing.assert_allclose(result.best_probability, [5 / 6, 1 / 6, 0])
assert result.global_enrolled == 10
assert result.effective_exponent == 0.25  # all 10 assignments / (2 * 20)
assert result.allocation_probability[2] == 0
print(result.allocation_probability)  # approximately [0.599254, 0.400746, 0]
```

Simply discarding a closed arm's probability from an earlier all-arm posterior
would give a different answer. Each call recomputes the beta posterior
best-arm probabilities using the active competitors. Successes and failures
must not exceed assigned counts; the remaining assignments can represent
pending outcomes. Every arm keeps its posterior shape and variance, including
closed arms, in the immutable result.

The four methods are BARCP, BARN2N, BARMTV and DBCD. Their equations and
parameters follow [BARPO](barpo.md). BARN2N requires the total trial budget
`max_n`, which must cover all historical assignments. DBCD requires explicit
positive targets and positive assigned counts for each active arm. Targets and
minimum allocation probabilities use full-ledger order and must be zero for
inactive arms. Targets are normalized within the active set. Floors use the
documented BARPO proportional water-filling policy; exact native behavior for
simultaneous experimental-arm floors has not been established.

The ledger supports up to 100 arms, with one to ten active competitors per
call. No random sampling or trial-sized intermediate array is needed. Invalid
counts, infeasible floors, nonfinite posterior shapes and numerical integration
failures are reported explicitly. An empty active set is a trial-stopping state
for a scheduler, not an allocation calculation.

`tools/reference_plbarpo_active.R` generates eight independent base-R integral
and allocation snapshots in `tests/fixtures/plbarpo-active-allocation.csv`.
They cover all four methods, floors, changes in the active set and a sole
remaining arm. Python posterior shapes, variances, best-arm probabilities,
allocation, global enrollment and exponent matched all eight scenarios. The
check took 0.039 seconds after import, peaked at 97.36 MiB and reported no swaps.
Two focused tests, lint, formatting and targeted type checks also passed.

This component does not choose when arms open or close. The separate
[no-control platform trial](plbarpo-trials.md) supplies explicit scheduling,
replacement, burn-in and complete-outcome replay. Control scheduling, aggregate
simulation and native reports remain open. See the
[platform source contract](../research/plbarpo-platform-audit.md) and existing
[control monitoring API](plbarpo-control.md).
