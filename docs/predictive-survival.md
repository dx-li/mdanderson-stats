# Predictive Probabilities: time-to-event outcomes

Catalog **10** now includes the survival model from chapter 2 of the
[version 1.5 guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/PredictiveProbabilit/PredictiveProbabilitiesUsersGuide.pdf),
alongside [binary interim analysis and planning](predictive-binary.md).
This is an independent Python implementation; native random-number streams and
undocumented boundary conventions are not reproduced. Source provenance is in
[predictive-probabilities-sources.json](predictive-probabilities-sources.json).

## Current-data comparison

```python
from mdanderson_stats import compare_predictive_survival

current = compare_predictive_survival(
    events=[10, 16],
    exposure=[150, 100],
    prior=[[1, 10], [1, 10]],
)
print(current.arm_a_probability, current.arm_b_probability, current.decision)
```

Events are undesirable, so larger mean survival is better. Prior rows are
`[shape,scale]` for independent inverse-gamma distributions on **mean survival**,
not median survival. Density is proportional to `mu**(-a-1)*exp(-b/mu)`.
After `d` events and total observed exposure `E`, the posterior is
`IG(a+d,b+E)`. An observed event contributes its event time; a censored patient
contributes follow-up to the current analysis.

If posterior shapes and scales are `(a_A,b_A)` and `(a_B,b_B)`, then
`P(mu_A>mu_B)=I_q(a_A,a_B)`, where `q=b_A/(b_A+b_B)`. Both ordering tails are
computed directly, using scaled ratios and log-space tail evaluation when a
ratio underflows. Leading input dimensions broadcast and the final axis holds
the two arms. Posterior shape, scale, both ordering probabilities, decision and
frequentist z statistic are returned. The z statistic is NaN in Bayesian mode.

The default Bayesian rule requires a probability strictly above
`posterior_cutoff=0.95`. `method="frequentist"` applies the guide's normal
approximation to the difference of estimated exponential hazards. Its
`significance_level=0.025` is **one-sided**, unlike the binary interface's
two-sided level. With positive exposure on both arms and no observed events,
the result is inconclusive. If either arm has zero exposure, the frequentist
comparison is `not_evaluable`; it is not silently assigned to either arm.

## Posterior predictive simulation

```python
from mdanderson_stats import predictive_survival

prediction = predictive_survival(
    patients=[25, 25],
    events=[10, 16],
    exposure=[150, 100],
    prior=[[1, 10], [1, 10]],
    accrual_rate=4,
    followup=6,
    max_patients=100,
    max_duration=30,
    elapsed_time=12,
    max_events=60,
    n_simulations=10000,
    rng=10,
)
# Order: A superior, B superior, inconclusive, not evaluable.
print(prediction.probability, prediction.mcse)
print(prediction.patients.mean(axis=0), prediction.events.mean(axis=0))
```

Each replication draws one mean survival for each arm from its current posterior,
then generates future exponential event times conditional on those means.
Patients who have not yet experienced an event remain at risk. Exponential
memorylessness allows their remaining event times to be simulated from aggregate
current counts and exposure; prior individual follow-up times need not be supplied.
This assumes administrative censoring at the interim, not permanent dropout.

New patients arrive according to a Poisson process with total `accrual_rate`.
Independent allocation assigns each new patient to A with
`allocation_probability`, default `0.5`, and otherwise B. Accrual stops at the
first enabled limit:

- `max_patients`: total current plus newly enrolled patients, across both arms.
- `max_duration`: elapsed accrual time plus future accrual time; the remaining
  duration is `max_duration-elapsed_time`.
- `max_events`: total current plus newly observed events, across both arms.

At least one limit is required. Limits equal to current values stop accrual
immediately, a useful extension for follow-up-only prediction. An event-driven
stop ends **accrual**, then `followup` continues for enrolled patients; the final
event count may exceed `max_events`. Events at the analysis time count as
observed. Enrollment at a stopping time is included; exact ties have probability
zero under the continuous simulation model.

Final exposure includes current exposure, further observation of currently
pending patients, and observation of newly enrolled patients. Exposure after an
event is excluded. The guide's Bayesian or frequentist comparison is then
applied to each completed simulated trial. `accrual_stop_time` and `final_time`
are measured forward from the current analysis, not from the original trial
start. Per-trial arm counts, events, exposure and decisions are retained.

The result reports four empirical probabilities and plug-in binomial Monte Carlo
standard errors. They are simulation estimates, not exact predictive
probabilities; an estimated standard error of zero does not establish certainty.
The `not_evaluable` category is relevant when a simulated frequentist comparison
lacks exposure on an arm. Bayesian comparisons remain defined through proper
priors. These estimates are conditional on the assumed model and current data,
not frequentist error-control guarantees.

The simulation draws future patients in chunks of up to 64. Extra generated
patients beyond the first stop are discarded before exposure is accumulated.
`max_simulated_patients`, default 10,000, bounds generated patients per replication
including current patients; reaching this resource guard before resolving a stop
raises an error rather than truncating the trial. Both it and `n_simulations`
allow at most 100,000. Priors and numeric time scales must be representable.

Validation checks analytic inverse-gamma ordering, a scale ratio spanning 600
orders of magnitude, time-unit invariance, the guide's hazard statistic, competing
calendar stops, exposure after censoring, and an analytic posterior-predictive
pending-event probability. The package supplies numerical results rather than
the desktop report-history interface or its separate Parameter Solver and
Inequality Calculator utilities.
