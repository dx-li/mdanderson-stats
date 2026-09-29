# Randomized binary comparisons in BOP2-DC

`bop2_dc_randomized_binary_design` implements the independent two-arm model
in §2.4 of the [primary paper](https://arxiv.org/abs/2112.10880).
It compares the experimental response probability minus the control response
probability with two signed clinical margins. Each arm has its own Beta prior.

```python
from mdanderson_stats import bop2_dc_randomized_binary_design

design = bop2_dc_randomized_binary_design(
    4, theta_lrv=0, theta_cmv=.2,
    control_prior=[1, 1], treatment_prior=[1, 1],
    arm_assignments=[0, 1, 0, 1], looks=[2, 4],
    lambda_lrv=.6, lambda_cmv=.5, gamma_lrv=.5, gamma_cmv=.5,
    graduate_at_interim=True,
)
trial = design.replay([0, 1, 0, 1])
print(trial.terminal_decision, len(trial.responses_observed))

oc = design.operating_characteristics([.2, .2], [.2, .7])
print(oc.graduate.sum(axis=-1) + oc.final_go)
print(oc.no_go_probability, oc.expected_sample_size)
```

This small example demonstrates conduct and exact recursion; its uncalibrated
cutoffs are not a recommended clinical design.

## Allocation and monitoring

The allocation tape uses 0 for control and 1 for experimental treatment, with
one assignment for every potential patient. Looks refer to cumulative total
enrollment. Priors, assignment tape and looks are explicit. The tape may be
unequal or produced by a separate randomization procedure. Operating
characteristics are conditional on this supplied allocation; the API does
not synthesize assignments or average over random allocation schedules.

`design.monitor(control_responses, control_sample_size, treatment_responses,
treatment_sample_size)` returns both posterior difference probabilities,
quadrature error estimates and a decision. Arm sample sizes must match the
allocation prefix. Between scheduled looks it returns `continue`. Monitoring
alone does not retain an earlier stopping decision.

At an interim look, both probabilities must fall strictly below their scaled
cutoffs for no-go. With `graduate_at_interim=True`, both probabilities must
strictly exceed their O’Brien–Fleming cutoffs for early graduation. For each
final cutoff `lambda`, the graduation boundary is
`2*Phi(Phi_inverse((1+lambda)/2)/sqrt(n/N)) - 1`. The implementation uses the
equivalent error-function expression to preserve tiny positive cutoffs.

At the final look, both probabilities must strictly exceed their cutoffs for
go or strictly fall below them for no-go. Other combinations, including exact
equality, produce consider. The LRV and CMV are differences, not arm response
rates; `-1 <= theta_lrv < theta_cmv <= 1` is required.

`design.replay(responses)` consumes a complete potential-response tape aligned
with the allocation. It returns only reached analysis states and observed
assignments/responses, stopping at the first no-go or graduation decision.

## Operating characteristics and numerical limits

The exact recursion propagates Bernoulli mass over the two arm response
counts, absorbing early no-go and graduation. `stop_no_go` and `graduate`
retain the look axis; their final entries are zero. Final go/consider/no-go
exclude early stopping. Add early graduation to final go for total favorable
decision probability. `no_go_probability` includes both early and final no-go.
`sample_size_probability` includes all terminal decisions at each look.

Independent Beta-difference tails are calculated by bounded quadrature.
Before classifying, the code checks all four corners of the two reported
probability/error intervals. It raises `ArithmeticError` if those estimates
could change the combined decision. One uncertain individual comparison can
still yield a stable combined action. The quadrature errors are estimates,
not rigorous bounds. Exact zero-error symmetry retains equality behavior.

Maximum enrollment is 1,000. Additional caps limit monitoring to 1,000 cells,
OC truth scenarios to 1,000, posterior comparisons to 5,000 count/margin cells,
recursion work to five million units and retained OC results to 100,000 cells.
These bounds are checked before the expensive tables or recursion allocations.
Posterior comparisons can make practical OC sizes smaller than the monitoring
limit. Outcomes must be completely observed; there is no delayed-data protocol.

Independent base-R integration and exhaustive four-patient path enumeration
verify posterior tails, strict decisions, early graduation, unequal allocation
and operating characteristics. See the
[audit](../research/bop2-dc-randomized-binary-audit.md).

## Finite-grid calibration

`optimize_bop2_dc_randomized_binary` searches the four lambda/gamma grids with
exact operating characteristics conditional on the allocation tape. Both truth
pairs are explicit `(control_probability, treatment_probability)` values;
control rates may differ between the futile and effective scenarios. The
futile treatment-control effect must be smaller than the effective effect,
and the effective effect must meet CMV.

```python
from mdanderson_stats import optimize_bop2_dc_randomized_binary

calibrated = optimize_bop2_dc_randomized_binary(
    4, theta_lrv=0, theta_cmv=.2,
    futile_truth=(.2, .2), effective_truth=(.1, .7),
    control_prior=(1, 2), treatment_prior=(2, 1),
    arm_assignments=(0, 1, 0, 1), looks=(2, 4),
    lambda_lrv_grid=(.6, .8), lambda_cmv_grid=(.2, .4),
    gamma_lrv_grid=(0, .5), gamma_cmv_grid=(0, .5),
    false_go_limit=.87, false_no_go_limit=.05, false_consider_limit=.2,
    graduate_at_interim=True, objective="cgr",
)
print(calibrated.selected_index)
print(calibrated.candidates.expected_sample_size[:, calibrated.selected_index])
```

This tiny, deliberately permissive example illustrates selection, not clinical
error targets. `objective="cgr"` maximizes effective-truth favorable decisions,
then minimizes futile expected enrollment; `"ess_futile"` reverses those
priorities. Favorable decisions include early graduation and final go. False
no-go includes interim and final no-go at the effective truth; optional false
consider is the larger final-consider probability across the two truths.
No feasible candidate raises `BOP2DCRandomizedBinaryInfeasibleError`.

`candidates.parameters` has columns lambda-LRV, lambda-CMV, gamma-LRV,
gamma-CMV in Cartesian product order (last grid varies fastest).
`decision_probability` has axes `(truth, candidate, look, decision)`;
`sample_size_probability` retains `(truth, candidate, look)`, and
`expected_sample_size` is `(truth, candidate)`. Truth order is futile then
effective. All candidates, including infeasible ones, remain visible when a
selection is available. Arrays are owned and read-only.

Posterior comparison tables are cached once per count state and reused across
the grid. Roundoff-level score ties retain input order, using a relative
binary64 tolerance scaled by trial length with no absolute floor. The same
small tolerance preserves inclusive error limits. It does not relax the strict
posterior decision rules or their quadrature safeguards. These conditional
exact probabilities have no Monte Carlo standard errors or holdout simulation.
Grid size, comparison tables, recursion work and live memory are bounded before
computation; exceeding a bound raises rather than silently approximating.

Independent R integration and exhaustive response-path enumeration validate
66 candidate configurations, both selection objectives, tied candidates and
infeasibility: 1,980 probability/enrollment summaries agree. See the
[calibration audit](../research/bop2-dc-randomized-binary-calibration-audit.md).
