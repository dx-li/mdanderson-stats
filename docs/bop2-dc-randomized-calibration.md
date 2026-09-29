# Calibrating randomized Normal and survival BOP2-DC designs

`optimize_bop2_dc_randomized_normal` and
`optimize_bop2_dc_randomized_survival` select from explicit finite grids of the
four lambda/gamma parameters. They apply the randomized extension in §2.4 of
the [BOP2-DC paper](https://arxiv.org/abs/2112.10880), using its CGR and futile
expected-enrollment objectives. For a binary outcome, use the
[exact randomized binary calibrator](bop2-dc-randomized-binary.md#finite-grid-calibration).

Supply a base design containing the priors, two margins, fixed allocation tape,
monitoring schedule, graduation setting and numerical tolerance. Both truth
scenarios are explicit `(control, treatment)` pairs. The futile signed
treatment-control effect must be below the effective effect, and the effective
effect must be at least CMV. The caller-declared futile effect may exceed LRV.
Normal outcomes additionally require both arm SDs for each scenario. Survival
truths are positive exponential medians, with explicit accrual and final follow-up.

## Normal example

```python
from mdanderson_stats import (
    bop2_dc_randomized_normal_design,
    optimize_bop2_dc_randomized_normal,
)

normal = bop2_dc_randomized_normal_design(
    4, theta_lrv=0, theta_cmv=.5,
    control_prior=(0, 1, 2, 1), treatment_prior=(.25, .75, 2.5, 1.5),
    arm_assignments=(0, 1, 0, 1), looks=(2, 4), graduate_at_interim=True,
)
fit = optimize_bop2_dc_randomized_normal(
    normal, futile_truth=(.25, .25), effective_truth=(-.25, 1),
    futile_truth_sd=(.75, 1.25), effective_truth_sd=(.5, 1),
    lambda_lrv_grid=(.4, .65), lambda_cmv_grid=(.2, .45),
    gamma_lrv_grid=(0,), gamma_cmv_grid=(.5,),
    false_go_limit=1, false_no_go_limit=1,
    n_trials=6, n_validation=4, rng=1560929,
)
print(fit.selected_index, fit.validation_feasible)
print(fit.validation_oc.decision_probability)
```

`fit.candidates.parameters` has columns lambda-LRV, lambda-CMV, gamma-LRV,
gamma-CMV. Candidate probabilities and their Monte Carlo standard errors
have shape `(2, candidates, 5)`; enrollment means/MCSEs have shape
`(2, candidates)`. Truth order is futile, effective. `calibration_oc` describes
the selected candidate on the calibration sample; `validation_oc` describes
its independent holdout and records its separate seed. Derived go/no-go rates
combine integer event counts before division, preserving exact Monte Carlo ties.

## Survival example

```python
from mdanderson_stats import (
    bop2_dc_randomized_survival_design,
    optimize_bop2_dc_randomized_survival,
)

survival = bop2_dc_randomized_survival_design(
    4, median_lrv=0, median_cmv=.5,
    control_prior=(2, 1.5), treatment_prior=(1.5, 1),
    arm_assignments=(0, 1, 0, 1), looks=(2, 4), graduate_at_interim=True,
)
fit = optimize_bop2_dc_randomized_survival(
    survival, futile_truth=(1, 1), effective_truth=(1.5, 3.5),
    lambda_lrv_grid=(.35, .65), lambda_cmv_grid=(.2, .45),
    gamma_lrv_grid=(0,), gamma_cmv_grid=(.5,),
    accrual_rate=2, final_followup=1.5, arrival="poisson",
    false_go_limit=1, false_no_go_limit=1,
    n_trials=6, n_validation=4, rng=1560930,
)
print(fit.selected_index, fit.validation_feasible)
print(fit.validation_effective.mean_enrollment)
```

Survival candidate counts/probabilities/MCSEs are available separately as
`futile_decision_*` and `effective_decision_*`, with shape `(candidates, 5)`.
The `grid` retains the original four grids; Cartesian product order has the last
grid varying fastest. Derived rate and enrollment arrays retain all candidates.
`validation_futile` and `validation_effective` include terminal decisions,
enrollment and arm-level event/exposure means and MCSEs. See the
[survival guide](bop2-dc-randomized-survival.md) for as-of censoring and clocks.

Both examples use tiny simulation samples and permissive limits to demonstrate
the interface. They are not calibrated clinical protocols.

## Selection, validation and limits

`objective="cgr"` maximizes effective-truth graduation plus final-go probability,
then minimizes futile expected enrollment. `"ess_futile"` minimizes that
expected enrollment, then maximizes correct-go probability. Remaining ties
retain input order. False-go uses graduation plus final go at the futile truth;
false-no-go uses interim plus final no-go at the effective truth. Optional
`false_consider_limit` constrains the larger final-consider probability across
the two truths. The decision order is `stop_no_go`, `graduate`, `final_go`,
`final_consider`, `final_no_go`.

Candidates share simulated paths. Posterior tails and numerical error estimates
are computed once per truth, trial and look, then reused across the grid. The
selected candidate is evaluated on an independent seeded holdout without
reselection. `validation_feasible=False` is a valid reported result, not an
instruction to search the holdout for a replacement. Empirical feasibility does
not guarantee true error control. Per-category probability MCSEs are binomial
plug-in estimates; enrollment MCSE uses the sample SD and is undefined for one
trial. Survival additionally exposes MCSEs for combined rates and each
final-consider component; no standard error is claimed for their maximum.

Numerical uncertainty that can change an active trial decision raises an error.
Candidate grids, path storage, quadrature work and peak live results are bounded
before consuming RNG state. Calls run serially; the default 100 trials per stage
is a workload choice, not a precision recommendation. Normal paths are centered
to preserve variation under common shifts. Survival calibration uses bounded
chunks and validation uses serial trial draws; native RNG parity is not claimed.

Independent R Student-t/Gamma density integration and complete trial replay
check candidate metrics and independent holdouts. The Normal reference includes
a calibration-pass/holdout-fail example without reselection. See the
[Normal audit](../research/bop2-dc-randomized-normal-calibration-audit.md) and
[survival audit](../research/bop2-dc-randomized-survival-audit.md).
