# Accelerated titration for BOIN combinations

`simulate_boin_combination(..., titration=True)` adds the accelerated-titration
workflow from CRAN BOIN 2.7.2. The trial initially treats one patient at each
visited combination. After a non-DLT outcome it increases one drug by one
level, choosing each direction with equal probability when both are available.
At an edge, it follows the remaining direction. Titration ends at the first
DLT or after treating the upper-right combination.

```python
import numpy as np
from mdanderson_stats import BOINCombDesign, simulate_boin_combination

result = simulate_boin_combination(
    BOINCombDesign(early_stop_patients=None),
    np.zeros((2, 2)),  # A simple all-safe reference scenario.
    cohorts=5,
    cohort_size=3,
    trials=10,
    titration=True,
    rng=128,
)
assert np.all(result.titration_patients.sum(axis=(1, 2)) == 3)
assert np.all(result.titration_endpoint == (2, 2))
assert np.all(result.patients.sum(axis=(1, 2)) == 17)
```

The first ordinary cohort fills the titration endpoint to `cohort_size`.
`cohorts` counts ordinary cohorts, so earlier staircase patients add to the
usual cohort total. Before stopping rules, the maximum enrollment is
`cohorts * cohort_size + (rows - start_row) + (columns - start_column)`.
The 1,000-patient limit includes this extra enrollment. A cohort size of one
disables titration, following the original simulator.

The result retains the usual total patient/toxicity matrices and final MTD
selection, plus immutable `titration_patients` and one-based
`titration_endpoint` arrays. `titration_end_reason` is `first_dlt`,
`upper_right`, `cohort_size_one`, or `not_requested`. When titration is
inactive its count matrix and endpoint are zero. Ordinary safety monitoring,
convergence stopping and final selection continue after the transition.
R and NumPy random streams differ; matching seeds do not imply matching paths.

The simulator rejects an estimated retained/working state above 128 MiB before
allocating trial arrays or drawing random values. This estimate includes
trial-level metadata and array copies; it is not a total-process RSS guarantee.

Nine deterministic cases execute the original R titration and first-cohort
expressions and agree exactly with Python. A further 100 simulated trials
without titration exactly reproduce the previously published results. The
[source and validation audit](../research/boin-combination-titration-audit.md)
records the remaining distinctions. App-specific moderate-toxicity stopping,
its separate titration cap, 3+3 run-in and waterfall titration remain open.
