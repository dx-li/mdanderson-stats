# Paired BOP2-DC calibration

`optimize_bop2_dc_paired` selects a design from an explicit finite grid using
exact operating characteristics. It supports both multiple efficacy endpoints
and efficacy/toxicity, including dependence between each patient's outcomes.
The objectives follow §2.3 of the [primary paper](https://arxiv.org/abs/2112.10880).

Provide two joint outcome distributions, in the order **both events, first
only, second only, neither**. For efficacy/toxicity, the second event is
toxicity. Marginal rates alone do not determine the joint distribution.

```python
from mdanderson_stats import optimize_bop2_dc_paired

result = optimize_bop2_dc_paired(
    4,
    "multiple_efficacy",
    lrv=[0.2, 0.15],
    cmv=[0.5, 0.45],
    futile_probabilities=[0.04, 0.16, 0.16, 0.64],
    effective_probabilities=[0.3, 0.3, 0.2, 0.2],
    prior=[0.3, 0.2, 0.4, 0.1],
    looks=[2, 4],
    lambda_lrv_grid=[[0.4, 0.6], [0.8, 0.9]],
    lambda_cmv_grid=[[0.15, 0.25], [0.45, 0.55]],
    gamma_lrv_grid=[[0, 0.5], [0.5, 0]],
    gamma_cmv_grid=[[0.5, 0.5]],
    false_go_limit=0.3,
    false_no_go_limit=0.7,
    false_consider_limit=0.9,
    objective="cgr",
)
i = result.selected_index
assert result.feasible[i]
print(result.design.lambda_lrv, result.design.lambda_cmv)
print(result.false_go_rate[i], result.correct_go_rate[i])
```

This four-patient example demonstrates the interface and does not prescribe
clinical thresholds or error limits. The effective truth must satisfy the
clinical-go composition: at least one efficacy margin at or above its CMV for
multiple efficacy; efficacy at or above CMV and toxicity at or below CMV for
efficacy/toxicity. The futile truth is explicitly supplied by the caller.

## Grids, objectives and evidence

A one-dimensional grid such as `[.7, .8]` supplies two candidates, each using
the same value for both endpoints. A matrix such as `[[.7, .8]]` supplies one
candidate with different endpoint values. This convention applies to all four
grids. `result.grid` retains the supplied control rows; candidate results use
their Cartesian product in the order lambda-LRV, lambda-CMV, gamma-LRV,
gamma-CMV, with the last grid varying fastest.

False-go rate is final go under the futile truth. False-no-go rate includes
early and final no-go under the effective truth. Correct-go rate is final go
under the effective truth. False-consider rate is the larger final-consider
probability across the two truths. Constraints include equality.

`objective="cgr"` maximizes correct-go probability, breaking ties by smaller
futile expected sample size. `objective="ess_futile"` reverses that priority.
Remaining ties use input/product order. `BOP2DCPairedInfeasibleError` means no
supplied candidate meets the constraints; it says nothing about other grids.

The result retains every candidate's error rates, feasibility, exact outcome
probabilities, stopping distribution and expected sample size for both truths.
`futile_oc.stop_no_go` and `effective_oc.stop_no_go` have candidate and look
axes. Final no-go excludes early stops. The final entry of each sample-size
distribution includes all three final decisions. There is no simulation error
or holdout stage: results are exact up to floating-point arithmetic for the
supplied joint truths and complete-outcome protocol.

## Computation and scope

Candidates are evaluated serially using the [paired recursion](bop2-dc-paired.md).
Each two-truth recursion requires `2 * (max_subjects + 1)**3 <= 5_000_000`.
The default combined `max_work` is 50 million units, with a hard ceiling of
100 million; candidate count and retained arrays are also bounded before OC
allocation. No full candidate-by-patient-path tensor is formed.

This implements finite-grid calibration, with explicitly declared tie rules.
It does not reproduce an undocumented app optimizer or delayed-outcome
protocol. Independent enumeration and validation details are recorded in the
[audit](../research/bop2-dc-paired-calibration-audit.md).
