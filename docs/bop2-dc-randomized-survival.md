# Randomized survival outcomes in BOP2-DC

`bop2_dc_randomized_survival_design` uses independent exponential event-time
models for the control and experimental arms. Following §2.4 of the
[primary paper](https://arxiv.org/abs/2112.10880), it compares experimental
median survival minus control median survival with two signed time margins.

```python
import numpy as np
from mdanderson_stats import (
    bop2_dc_randomized_survival_design,
    run_bop2_dc_randomized_survival_trial,
    simulate_bop2_dc_randomized_survival,
)

design = bop2_dc_randomized_survival_design(
    4, median_lrv=0, median_cmv=.5,
    control_prior=[2, 1], treatment_prior=[2, 1],
    arm_assignments=[0, 1, 0, 1], looks=[2, 4],
    lambda_lrv=.5, lambda_cmv=.5, graduate_at_interim=True,
)
trial = run_bop2_dc_randomized_survival_trial(
    design, enrollment_times=[0, .25, .5, .75],
    event_times=[.5, np.inf, .25, np.inf], final_followup=.5,
)
print(trial.decision, trial.enrolled)
oc = simulate_bop2_dc_randomized_survival(
    design, control_true_median=2, treatment_true_median=3,
    accrual_rate=4, final_followup=.5, n_trials=6, rng=419,
)
print(dict(zip(oc.decision_labels, oc.decision_probability)))
print(oc.mean_enrollment, oc.mean_events)
```

These uncalibrated settings and small simulation count illustrate the API.

## Model and monitoring

Each explicit prior is `IG(shape, scale)` on the arm's **mean** survival.
Both parameters must be positive. With `d` events and total exposure `t`, the
posterior is `IG(shape+d, scale+t)`; exponential median survival is `log(2)`
times the mean. A zero-duration event increases the shape while contributing
zero exposure. Censoring contributes exposure without an event.

The design requires a fixed 0/1 allocation tape and total-enrollment looks.
Unequal allocation is supported. `design.monitor(control_events,
control_exposure, control_n, treatment_events, treatment_exposure, treatment_n)`
accepts scalar or bounded broadcasting inputs. Each arm's sample size must
match the corresponding allocation prefix. An empty arm has zero events and
exposure, retaining its prior.

States include arm counts/events/exposure, posterior shape and scale with
last axis `[control, treatment]`, both posterior difference probabilities,
error estimates and actions. Signed margins are differences in median time;
they are not survival ratios or hazard ratios. A zero margin uses an analytic
Beta ratio identity. Nonzero margins use dimensionless Gamma quadrature, so
changing time units also requires scaling prior scales and both margins.

Interim no-go, optional O'Brien-Fleming graduation and final go/consider/no-go
use the [shared randomized rules](bop2-dc-randomized-binary.md). Equality does
not satisfy a strict stopping inequality. Numerical uncertainty that could
change an action raises an error; quadrature errors are estimates rather than
rigorous bounds.

## Calendar replay and simulation

Replay accepts ordered enrollment times and event **durations from enrollment**.
Positive infinity means no observed event. At an interim look, follow-up ends
at the enrollment time of that look's last patient. At the final look, add
`final_followup` after the final enrollment. Events exactly on the analysis
boundary count; input order breaks enrollment ties. Pending event durations
contribute censored exposure. The first no-go or graduation stops the replay.

Simulation requires both true medians, accrual rate and final follow-up.
`arrival="fixed"` enrolls at `1/rate, 2/rate, ...`; `arrival="poisson"` draws
independent exponential arrival gaps. Event durations are independent exponentials
with arm-specific mean `true_median/log(2)`. The fixed allocation tape is reused
for every trial. The default 100 trials is a Python workload choice; simulations
run serially with bounded storage. Reusing the returned seed and arguments
reproduces the result in the same numerical environment.

Results retain compact per-trial enrollment, arm enrollment/events/exposure,
calendar duration and terminal decisions, together with aggregate decision
probabilities and Monte Carlo errors. Arm summaries use `[control, treatment]`
order. Graduation and final go are separate categories; add them for total
favorable probability. Calendar duration is measured from time zero, including
waiting for the first enrollment. MCSEs for continuous summaries require at
least two trials. Arrays are owned and read-only.

Before posterior work or random generation, explicit bounds check batch cells,
calendar scans, retained storage and cumulative quadrature effort. A replay
also has its own total quadrature-work limit. Maximum enrollment is 1,000;
the practical workload can be smaller when many looks or nonzero margins
require integration. These are resource limits, not clinical design defaults.

Independent R Gamma-density integration and as-of calendar calculations provide
validation; see the [audit](../research/bop2-dc-randomized-survival-audit.md).
Randomized survival calibration remains open.
