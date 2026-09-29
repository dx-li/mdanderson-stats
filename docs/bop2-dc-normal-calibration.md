# BOP2-DC Normal endpoint calibration

`optimize_bop2_dc_normal` calibrates the [continuous Normal design](bop2-dc-normal.md)
on an explicit finite parameter grid, following §2.3 of the
[primary paper](https://arxiv.org/abs/2112.10880). It uses common simulated
observations across candidates and independent observations for validation.

```python
from mdanderson_stats import optimize_bop2_dc_normal

result = optimize_bop2_dc_normal(
    8, theta_lrv=0, theta_cmv=.5, theta_futile=-.5, theta_effective=1.5,
    truth_sd=1.3, prior_mean=0, prior_precision=.5,
    prior_shape=1.5, prior_scale=.75, looks=[2, 4, 8],
    lambda_lrv_grid=[.6, .8, .95], lambda_cmv_grid=[.2, .5, .8],
    gamma_lrv_grid=[0, .5], gamma_cmv_grid=[.5],
    false_go_limit=.04, false_no_go_limit=.2, false_consider_limit=.8,
    objective="cgr", n_trials=64, n_validation=48, rng=1560932,
)
i = result.selected_index
assert result.candidates.feasible[i]
assert not result.validation_feasible
print(result.candidates.parameters[i])
print(result.candidates.false_go_rate[i], result.validation_false_go_rate)
```

The selected design has an estimated false-go rate of `2/64` during calibration
and `3/48` on the holdout. It passes the calibration limit of 4% and fails
validation. The result preserves this failure and does not select a different
design using holdout outcomes. The small trial counts demonstrate the API,
not a precise assessment of clinical error rates.

## Inputs and objectives

Supply both truth means, a positive `truth_sd` shared by the two scenarios,
all four proper NIG prior parameters, the clinical thresholds and four
one-dimensional cutoff/exponent grids. The futile mean must be smaller than
the effective mean, and the effective mean must be at least CMV. Signed means
and clinical thresholds are supported. Outcome variance remains unknown in
the posterior model even though simulation requires a truth SD.

False-go rate is final go under the futile truth. False-no-go rate includes
both early and final no-go under the effective truth. Correct-go rate is final
go under the effective truth. False-consider rate is the larger final-consider
probability across both truths. Error constraints include equality.

`objective="cgr"` maximizes correct-go rate, then minimizes futile expected
sample size. `objective="ess_futile"` reverses that priority. Remaining ties
use input/product order. If the supplied grid has no feasible candidate,
`BOP2DCNormalInfeasibleError` reports that result. These are empirical Monte
Carlo constraints; they do not establish control of the true error rates.

## Results and reproducibility

`result.candidates.parameters` has one row per candidate and columns
`lambda_lrv`, `lambda_cmv`, `gamma_lrv`, `gamma_cmv`, with the final grid varying
fastest. Candidate decision probabilities and their MCSEs have shape
`(2, candidates, 4)`: futile/effective truth, candidate, then the four labels
in `result.calibration_oc.decision_labels`. Enrollment means and MCSEs have
shape `(2, candidates)`. Error rates, correct-go rates and feasibility retain
the candidate axis. All output arrays are owned and read-only.

`calibration_oc` describes the selected candidate. `validation_oc` describes
its independent holdout, with the same scenario and decision ordering.
`validation_*_rate` and `validation_feasible` report the holdout verdict.
Scenario-specific consider MCSEs remain separate; the maximum of two consider
rates does not have a binomial MCSE.

`rng_seed` recreates the entire call in the same numerical environment.
`calibration_oc.rng_seed` and `validation_oc.rng_seed` identify the independent
stage streams. Each stage shares standardized Normal draws across both truths
and all candidates. Calculations remain relative to the truth mean, preserving
variation when all model locations have a large common offset.

Serial vectorized batches bound temporary arrays. Limits are 10,000 candidates,
100,000 trials per stage, one million potential patient draws across both
stages, 50 million path/candidate work units and two million array cells for
combined result/workspace storage. Batch width adapts to the remaining space;
the calibration path matrix is released before allocating holdout paths.

An independent base-R replay verifies all 18 candidates, both selected designs
and their holdouts, including the failed validation shown above. See the
[source and numerical audit](../research/bop2-dc-normal-calibration-audit.md).
