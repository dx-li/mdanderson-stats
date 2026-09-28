# BOP2-DC survival design calibration

`optimize_bop2_dc_survival` selects a survival design on an explicit finite
grid, following the objectives in §2.3 of the BOP2-DC
[primary paper](https://arxiv.org/abs/2112.10880). It uses the
[exponential survival monitor and calendar protocol](bop2-dc-survival.md),
shared simulation paths across candidates, and an independent validation stage.

Supply both truth medians, the clinical thresholds, all four parameter grids,
the inverse-gamma mean prior, analysis looks, accrual and follow-up settings,
and the desired error limits. The effective truth must be at least CMV; the
futile truth must be positive and smaller than the effective truth.

```python
from mdanderson_stats import optimize_bop2_dc_survival

result = optimize_bop2_dc_survival(
    6, lrv=3, cmv=5, theta_futile=1.5, theta_effective=8,
    lambda_lrv_grid=[.4, .7, .9], lambda_cmv_grid=[.1, .3, .5],
    gamma_lrv_grid=[0, .5], gamma_cmv_grid=[.5],
    prior_shape=1, prior_scale=2, looks=[2, 4, 6],
    accrual_rate=1, final_followup=4, arrival="poisson",
    false_go_limit=.04, false_no_go_limit=.6, false_consider_limit=.8,
    objective="cgr", n_trials=64, n_validation=48, rng=1560930,
)
i = result.selected_index
assert result.feasible[i]
assert not result.validation_feasible
print(result.design, result.validation_false_go_rate)
```

This small example deliberately illustrates a design that passes the
calibration sample and fails the independent check: estimated false-go risk
is `2/64` during selection and `3/48` during validation, against a 4% limit.
The result retains the selected design and reports failure; it does not
search again using the holdout data. These small simulation counts demonstrate
the interface and do not establish a practically precise trial design.

## Objectives and returned evidence

| Metric | Definition |
| --- | --- |
| False-go rate, FGR | Final go probability at the futile truth |
| False-no-go rate, FNGR | Early plus final no-go probability at the effective truth |
| Correct-go rate, CGR | Final go probability at the effective truth |
| False-consider rate, FCR | Larger final-consider probability across both truths |

`objective="cgr"` maximizes CGR, then prefers lower futile expected sample
size. `objective="ess_futile"` minimizes futile expected sample size, then
prefers higher CGR. Remaining ties select the first candidate in the product
order of `lambda_lrv`, `lambda_cmv`, `gamma_lrv`, and `gamma_cmv`; each input
grid retains its supplied order. Constraints include equality. If no candidate
passes, `BOP2DCSurvivalInfeasibleError` reports that finite-grid result.

The result retains every candidate's scenario decision probabilities and
binomial Monte Carlo errors, error/objective metrics, sample-size means and
errors, feasibility and selected index. Decision columns follow
`validation_futile.decision_labels`: early no-go, final go, final consider,
final no-go. Final and early no-go are disjoint; their sum defines FNGR.
Scenario-level consider errors are returned separately, since the maximum
defining FCR is not itself a binomial estimator.

`validation_futile`, `validation_effective`, the `validation_*_rate` fields
and `validation_feasible` expose the separate holdout results. Empirical limits
constrain simulation estimates. Neither a successful selection nor a passing
holdout guarantees that the true error probabilities satisfy the limits.

## Reproducibility and bounds

`rng_seed` replays the entire call in the same numerical environment.
`calibration_seed` and `validation_seed` identify separate stages. Within each
stage, both truth scenarios restart from the same seed, sharing standardized
event/accrual draws. All candidates use the same calibration paths. A supplied
Generator contributes one master seed. Parameter-grid, minimum work and
storage checks occur before randomness is consumed.

Computation is serial. Path chunks and candidate batches fit within explicit
memory caps; the full candidate-by-patient-by-trial array is never formed.
At most 10,000 candidates and 100,000 trials per stage are supported, subject
to combined storage and repeated-look work budgets. The default `max_work`
is 50 million units and the hard ceiling is 100 million. The simulator's
own validation-stage bounds also apply.

Independent base-R replay verifies all 18 candidates in the example, including
760 decision-probability, sample-size and Monte Carlo error summaries across
both objectives and their holdouts. The selected candidates differ between
objectives. A separate one-patient analytic calculation tests nonzero go,
consider and no-go regions plus the censoring point mass. See the
[source audit](../research/bop2-dc-survival-calibration-audit.md).
