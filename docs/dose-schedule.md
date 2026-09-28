# Dose Schedule Finder

This is an independent Python implementation of the method identified by
[Dose Schedule Finder 2.2.0](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/75):
Braun, Thall, Nguyen and de Lima (2007),
[Simultaneously optimizing dose and schedule of a new cytotoxic agent](https://odin.mdacc.tmc.edu/~pfthall/main/ClinTrials%20dose-sched%202007.pdf).

## Model and data

Each administration contributes a triangular toxicity hazard. For a dose,
`area` is its total integrated hazard, `peak` is time from administration to
maximum hazard, and `tail` is the remaining duration. Patient hazards and
cumulative hazards add across actual administrations. Risk is
`-expm1(-cumulative_hazard)`; event likelihoods include the hazard at the
observed time, while censored records contribute survival only.

Use one time unit consistently for follow-up, administration times and hazard
timing. Dose indices are zero-based categories; the model does not interpolate
between numerical dose amounts. Actual histories can include delays and dose
changes. Candidate regimens use one dose throughout a nested schedule.

## Priors

The normal coordinates are log area (or positive area increment), log peak and
log tail, separately at each dose. Ordered areas accumulate positive increments.
The elicitation helper accepts shortest-schedule toxicity probabilities and
mean hazard timings, then applies the paper's approximate moment equations.
The supplied toxicity probabilities are not an exact prior-predictive match:
the transformation is nonlinear and full hazard accumulation is assumed.
The two tuning constants exceed one and determine the prior variances.

For the paper's example, shortest-schedule probabilities are `.20, .25, .30`,
with five administrations, peak means `18, 14, 10`, tail means `10, 14, 18`,
and both tuning constants `1.5`. These give log variances `log(3)` and reproduce
the published rounded normal means. Explicit log-normal parameters are also
supported; a zero standard deviation fixes that coordinate.
The example's tuning constants were selected through study-specific simulation;
their default values do not constitute calibration for a new study.

```python
from mdanderson_stats import dose_schedule_moment_prior

prior = dose_schedule_moment_prior(
    [0.20, 0.25, 0.30],
    administrations=5,
    peak_times=[18, 14, 10],
    tail_times=[10, 14, 18],
    lambda1=1.5,
    lambda2=1.5,
)
print(prior.mean.reshape(3, 3))
```

Coordinates are ordered by dose, then log area increment, log peak, log tail.
Use `dose_schedule_parameter_names` to inspect the ordering for a particular
model. Disable `ordered_areas` when area increments are inappropriate; the
area coordinate then describes the dose's own area directly.

## Fit and select a regimen

This small example uses the prior above and two synthetic patient histories.
Short chains demonstrate the interface; assess the precision of posterior
risks and safety-tail probabilities before using them in an analysis.

```python
import numpy as np
from mdanderson_stats import (
    DoseSchedulePatient,
    fit_dose_schedule,
    dose_schedule_decision,
)

first_course = [0, 1, 2, 3, 4]
two_courses = first_course + [28, 29, 30, 31, 32]
patients = [
    DoseSchedulePatient(16, True, first_course, [0] * 5),
    DoseSchedulePatient(45, False, two_courses, [1] * 10),
]
fit = fit_dose_schedule(
    patients,
    prior,
    [first_course, two_courses],
    horizon=116,
    draws=128,
    warmup=64,
    chains=2,
    rng=np.random.default_rng(75),
)
print(fit.risk_summary.mean)
decision = dose_schedule_decision(
    fit,
    treated=[[1, 0], [0, 1], [0, 0]],
    toxicity_limit=0.30,
    upper_probability=0.80,
    target=0.30,
)
print(decision)
```

`fit.regimen_risk` has axes `(chain, draw, dose, schedule)`. Retained log
parameters and physical hazard parameters are available separately. Summaries
include split R-hat and batch-means Monte Carlo standard errors. A completed
sampling run does not establish adequate precision; `summarize_chains` can
also summarize a threshold indicator formed from the retained risks.

With no patients, the fit draws directly from the prior. It uses serial
elliptical slice sampling otherwise. An event outside every administration's
finite hazard support has zero likelihood and can reject a sampler proposal.
Unrepresentable arithmetic raises an error. Patient counts, administration
counts, retained arrays, likelihood calls and administration/prediction work
are bounded before or during computation. The fit permits at most 200 patients,
20 doses, 10,000 actual administrations, and 20 nested schedules with at most
20 administrations each (200 across all schedules). Schedule lengths must
increase, and each must contain its predecessor as a subsequence; administration
times need not form a prefix of the longer schedule.

Sampling uses 2–8 chains, 8–100,000 retained draws per chain and at most 100,000
warmup iterations. The combined retained parameter, physical-parameter,
risk and likelihood arrays cannot exceed two million cells. `max_evaluations`
defaults to 200,000 likelihood calls (hard maximum two million); `max_work`
defaults to 20 million administration/prediction contributions (hard maximum
50 million). Impossible minimum budgets are rejected before drawing random
numbers or allocating retained arrays. Exceeding a running budget raises an
error. These bounds limit a single fit; run fits serially when memory is limited.

## Posterior decisions

Safety requires `P(regimen_risk > toxicity_limit) < upper_probability`.
Equality at the outer cutoff is unsafe. Among eligible combinations, selection
minimizes the distance between mean risk and the target. The first assignment
defaults to the lowest dose and shortest schedule, bypassing the safety screen
for that assignment while still reporting the computed safety mask. `starting`
allows an explicit alternative as a Python extension. Subsequent escalation
cannot skip untried levels; final selection removes that restriction and
requires at least one treated patient. Exact ties use the lowest dose, then
shortest schedule.

For histories where neither tried pair dominates the other, the paper does not
fully specify the no-skip convention. The default `escalation_rule="coordinate_max"`
allows at most one level beyond the largest tried dose and largest tried
schedule, separately. `"observed_pair"` instead requires that a candidate be at
most one level above both coordinates of at least one previously tried pair.
Both permit diagonal one-level escalation, revisits and de-escalation. These
are documented Python conventions, not verified executable behavior. The
optional `current` pair is validated but does not change either history-based
rule.

The caller controls when a final analysis is performed. The paper's final
analysis follows completion of follow-up to the horizon, except for patients
who already had the toxicity event. Calling a selection function does not
establish that the trial has completed this requirement.

The Python implementation computes each regimen's risk directly. Increasing
total hazard area does not itself guarantee increasing risk at every finite
time when peak and tail durations vary. It therefore does not infer that every
higher dose is unsafe solely because a lower dose was excluded.

## Numerical evidence and remaining work

Independent base-R references integrate the administration hazard, combine
variable-dose histories, reproduce the published prior parameters and integrate
a reduced one-dimensional posterior. See the [audit](../research/dose-schedule-audit.md)
and [source record](dose-schedule-sources.json) for the validated scope.

The original Windows executable has not been run for equivalence. Full calendar
simulation, operating-characteristic calibration and native file/report
workflows remain open. The original executable and article are not bundled.
