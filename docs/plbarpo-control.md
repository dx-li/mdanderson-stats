# PLBARPO control monitoring

PLBARPO supports comparison with all trial controls or with controls concurrent
with each treatment. The Python API makes the contributing observations explicit.
Treatment arrays exclude the control arm.

```python
import numpy as np
from mdanderson_stats import plbarpo_control_counts, plbarpo_control_monitor

counts = plbarpo_control_counts(
    enrollment_time=[0, 1, 2, 3, 4, 5],
    outcomes=[1, 1, 0, 0, np.nan, 0],
    observation_time=[0.5, 1.5, 2.5, 3.5, 6, 5.5],
    windows=[[0, 4], [3, np.inf]],
    as_of=5.75,
)
result = plbarpo_control_monitor(
    successes=[3, 4], failures=[1, 2],
    prior=[[1, 1], [1, 1]], control_prior=[1, 1],
    control_counts=counts, control_mode="concurrent",
    pfut=0.25, peff=0.8, pfinal=0.7,
)
print(counts)  # [[2, 2], [0, 2]]: columns are successes, failures
print(result.efficacy_probability)  # approximately [0.738095, 0.916667]
```

Concurrent windows select controls by **enrollment time** using `[open, close)`.
Enrollment at a closing boundary is excluded. A positive-infinite close leaves
a window open. Both enrollment and outcome observation must occur by `as_of`.
A pending outcome may be `NaN` only while its observation time is later than
`as_of`; positive-infinite observation time represents an outcome with no
scheduled availability. If an outcome should already be available but is
missing, the helper raises an error.

For `control_mode="entire"`, pass one `(successes, failures)` pair shared by
all treatment comparisons. For `"concurrent"`, pass a matrix with one such row
per treatment, directly as returned by the selection helper. The control prior
is a single explicit Beta shape pair, updated separately for each comparison.
Shared or overlapping control observations can make the comparisons dependent;
the returned probabilities are marginal treatment-versus-control probabilities.

Futility compares `P(control > treatment)` strictly above `pfut`. Early and
final efficacy compare `P(treatment > control)` at or above `peff` and `pfinal`.
These are separate criteria for the caller's monitoring stage. The result
includes posterior shapes, direct tail probabilities, numerical error estimates
and boolean decision arrays.

Selection handles at most 100,000 control records and 100 windows, processing
one window at a time. Posterior monitoring supports up to 100 treatments.
No record-by-window matrix is allocated.

Automatic arm replacement, platform allocation, burn-in, trial scheduling,
simulation and reports remain pending. The guide's exact concurrency-boundary
convention is unverified; the explicit window contract above defines Python
behavior. See [source notes](plbarpo-control-source.md) and
[independent references](plbarpo-control-reference.md).
