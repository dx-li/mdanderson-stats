# Continuous normal and survival cutoff calibration

`calibrate_normal_success_cutoff` and `calibrate_survival_success_cutoff` find
a continuous success-probability cutoff whose probability of incorrect decision
(PID) is at most a requested target. The calibration uses the existing normal
and log-hazard-ratio operating-characteristic calculations; it adds no model,
prior or decision rule.

```python
from mdanderson_stats import (
    calibrate_normal_success_cutoff,
    calibrate_survival_success_cutoff,
)

normal = calibrate_normal_success_cutoff(
    target=0.05,
    standard_error=0.25,
    design_mean=0.3,
    design_sd=0.4,
    analysis_mean=0.0,
    analysis_sd=1.0,
    margin=0.0,
)
assert normal.operating_characteristics.incorrect_decision_probability <= 0.05
assert normal.bracket[1] - normal.bracket[0] <= normal.cutoff_tolerance

survival = calibrate_survival_success_cutoff(
    target=0.05,
    events=120,
    treatment_allocation=2 / 3,
    design_mean=-0.2,
    design_sd=0.25,
)
```

The default cutoff range `(0.6, 0.999)` follows the application's documented
calibration range. The `normal_success_oc` or `survival_success_oc` model
arguments are supplied directly as keyword arguments, excluding their cutoff.
The selected cutoff is the feasible upper endpoint of `bracket`, whose lower
endpoint is infeasible unless both endpoints are equal because the range's
lower endpoint already meets the target. The default bracket width is at most
`1e-10`. This bounds only the cutoff bracket width; it is not an error bound on
the operating-characteristic quadrature or the statistical target. The
underlying OCs retain their documented numerical accuracy and limitations.
`candidates_evaluated` counts operating-characteristic evaluations.

For the supported normal model, the truth and posterior decision statistic have
positive correlation, including the two-independent-arm model with positive
sampling and prior variances. Conditional probability of an ineffective truth
decreases as the posterior statistic moves farther into the favorable tail.
The survival adapter is the same normal model with the favorable tail reversed.
Thus PID is nonincreasing with cutoff and bisection can maintain an infeasible
lower endpoint and feasible upper endpoint. The implementation raises when the
upper endpoint misses the target, success probability is numerically unresolved,
or the evaluation cap is insufficient to reach the requested tolerance; it
never returns an unverified endpoint as a calibrated result.

This is a bounded Python bisection convention, not a reconstruction of the
application's optimizer, stopping rule or rounding. Existing
`calibrate_success_cutoff` remains a separate discrete candidate-grid search
for arbitrary evaluators and is unchanged.
