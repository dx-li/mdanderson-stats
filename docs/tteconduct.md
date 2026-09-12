# TTEConduct survival monitoring

Catalog 63 now provides the exponential/inverse-gamma stopping rule and
continuous time-on-test boundary table described in the
[TTEConduct guide](tteconduct-source.md).

```python
from mdanderson_stats import (
    tteconduct_design, tteconduct_monitor, tteconduct_boundary_table,
)

# Shape/scale priors on exponential MEAN survival. Time is in months here.
design = tteconduct_design(
    60, 295, 3, 10, 1, 0.03, 40, max_total_time=40 * 120,
)
state = tteconduct_monitor(design, patients=10, events=3, total_time=1)
print(state.probability, state.stop_accrual)
table = tteconduct_boundary_table(design, events=[1, 2, 3, 4, 5, 6])
print([row.minimum_total_time for row in table.boundaries])
```

The first two constructor arguments specify the historical prior; the next
two specify the experimental prior. `delta=1` requires an improvement of one
month in **mean survival**, and `cutoff=0.03` stops accrual if
`P(mu_experimental > mu_standard + delta | data) < 0.03`.
Experimental observations update only the experimental prior. Mean and median
are different: exponential median survival is mean survival times `log(2)`.
All scales, exposure, improvements, and the search cap must use consistent
units. For elicitation from moments or two quantiles, the package's
`solve_distribution_moments` and `solve_distribution_quantiles` support
the `inverse_gamma` family.

`total_time` is the sum of event times for patients with an observed event and
follow-up times for those still event-free. The monitor returns the posterior
ordering probability and estimated quadrature error, separate futility and
maximum-patient flags, and their combined `stop_accrual` flag. It evaluates the
current data; callers retain prior stop decisions and control subsequent
enrollment. It does not simulate arrival times or censoring records.

The table defaults to event counts 1 through `max_patients`; supplying a small
subset is useful for selected analyses. A zero boundary means that the prior
and that event count already satisfy the continuation rule at zero exposure.
A positive boundary solves the posterior probability equation. If the rule
is still unmet at `max_total_time`, the row has `beyond_cap=True` and an infinite
boundary; its probability and residual describe the search cap. This is an
unresolved boundary beyond the requested range, not proof that continuation is
impossible at every exposure.

Boundary roots are numerical approximations. Reported probability error and
residual help assess precision near the stopping cutoff; the quadrature error
is an estimate, not a rigorous floating-point bound. The strict monitor
comparison is not changed by a tolerance band. Zero improvement uses an
analytic beta identity; positive improvement uses one-dimensional quadrature.
Calculations run sequentially with scalar posterior comparisons and bounded
table sizes, without trial-by-patient simulation matrices.

The six-row guide example is checked against independent base-R integration
and root finding. See [source and reference details](tteconduct-source.md)
for the published rounded day values and the explicit conversion that
reproduces them. The Python table retains continuous values in the input
time unit. Native HTML report save/reopen behavior remains unimplemented;
calendar simulation belongs to the separate catalog 98 program.
