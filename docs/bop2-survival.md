# BOP2 single-arm survival monitoring

BOP2 catalog **112** now includes specified-parameter survival monitoring,
follow-up-time boundaries, calendar replay, and Monte Carlo operating
characteristics. Automatic survival calibration, survival sample-size searches,
two-arm/joint survival models and integrated reports remain pending.

The model is the exponential/inverse-gamma model described by
[Zhou et al. (2020), DOI 10.1002/pst.2030](https://pubmed.ncbi.nlm.nih.gov/32524679/).
The [current app prior guide](https://biostatistics.mdanderson.org/shinyapps/BOP2/TTEhelp.pdf)
and a [live app table capture](bop2-survival-app.json) establish its prior inputs
and the boundary's dependence on enrolled patients, events and total observation
time. Sources are recorded in [provenance](bop2-sources.json). This implementation
does not claim native optimizer or simulated-OC parity.

## Model and prior

For exponential survival with median `m`, the event rate is `log(2)/m`.
An event contributes its event time; a censored subject contributes observed
follow-up through the analysis time. With `d` events and total observation time
`T`, a prior `m ~ IG(a,b)` gives the posterior

$$
m\mid D \sim IG(a+d,b+\log(2)T).
$$

Here the inverse-gamma density is proportional to
`m**(-a-1) * exp(-b/m)`. Its scale `b` has time units.
The posterior probability of improving upon null median `m0` is the regularized
lower incomplete gamma function evaluated at `(b+log(2)*T)/m0`, with shape `a+d`.
This lower tail is computed directly.

The default prior follows the app guide's **elicitation equations**:
`a=1+prior_effective_events`, `b=prior_effective_events*m0`.
Thus the prior mean of the median survival parameter is `m0`; the default
effective event count is `0.05`. The guide also prints `IG(1.05,0.072*m0)` as a
default, which conflicts with those equations when interpreted as a prior on the
median. Approximately `0.072*m0` is the scale for the **mean-survival** parameter;
multiplication by `log(2)` converts that scale to approximately `0.05*m0` for the
median. This implementation uses the exact elicitation equations, not the rounded
and inconsistently labeled default sentence. An explicit `prior=[shape,scale]`
always refers to the **median** parameter and supersedes the ESS input.

The returned posterior scale is reported as `posterior_scale_ratio`, relative to
`null_median`. Keeping this dimensionless avoids multiplying large or small time
units unnecessarily. The same decisions should result after consistent changes
of time units.

## Monitoring

```python
from mdanderson_stats import bop2_survival_design

design = bop2_survival_design(
    30,
    null_median=6,
    cutoff_scale=0.85,
    gamma=0.85,
    looks=[15, 30],
)
state = design.monitor(events=8, total_observation_time=90, sample_size=15)
print(state.success_probability, state.decision)
print(design.total_time_boundary(events=[0, 5, 10, 15], sample_size=15))
```

These cutoff parameters are illustrative and **not automatically calibrated**.
At a scheduled look with `n` enrolled subjects, require posterior improvement
probability greater than `cutoff_scale*(n/N)**gamma`; otherwise stop for futility.
`equality_continues=False` follows the app's displayed `total time <= boundary`
stopping convention. Set it to `True` for strict-inequality futility instead.
At `N`, return `final_positive` or `final_negative`. There is no early efficacy
stopping; off-schedule evaluations return `continue`.

Both the event count and enrolled count are needed: censored patients contribute
follow-up without increasing the number of events. Inputs broadcast over leading
axes. `monitor_records(followup,event_observed)` aggregates observed patient data
on a final patient axis; event indicators must be zero or one. Do not pass latent
event durations for censored patients to this aggregation method.

`total_time_boundary` algebraically inverts the posterior rule for a specified
event/enrollment count. A negative threshold means no nonnegative follow-up time
can satisfy futility. The reported boundary is rounded floating-point output;
use `monitor` for decisions near it. Unrepresentable or underflowed numerical
quantities raise errors where they would invalidate a decision or reported
boundary. Monitoring supports up to 200 subjects, positive null median, scales
strictly in `(0,1)`, and `gamma` in `[0,1]`.

## Calendar replay and simulation

```python
from mdanderson_stats import run_bop2_survival_trial, simulate_bop2_survival

# Use the design defined above. Times below are durations from enrollment.
trial = run_bop2_survival_trial(
    design,
    enrollment_times=[(i + 1) / 1.5 for i in range(30)],
    event_times=[10] * 30,
    final_followup=12,
)
print(trial.calendar_times)
print(trial.states[-1].decision)
oc = simulate_bop2_survival(
    design,
    true_median=10,
    accrual_rate=1.5,
    final_followup=12,
    n_trials=10000,
    arrival="fixed",
    rng=960,
)
print(oc.success_probability, oc.success_mcse, oc.expected_sample_size)
```

Replay assesses each interim at the enrollment of that look's last patient.
The final analysis occurs at the last enrollment plus `final_followup`. All
subjects are administratively censored at each analysis; only patients enrolled
through that look are included. Replay stops immediately on the first negative
interim decision. Nonnegative latent event durations and `+inf` for no event are
accepted. Enrollment times must be nonnegative and ordered. Tied enrollment times
follow the supplied order of patients.

Simulation generates exponential event times. `arrival="fixed"` uses equally
spaced enrollment, while `"poisson"` uses exponential interarrival times. Both
include one interarrival interval before the first enrollment. Simulation uses
batches of up to 5,000 trials, supports up to 100,000 trials per call, and returns
per-trial sample sizes, event counts, calendar stopping times, and successes,
plus the empirical success probability, its Monte Carlo standard error, and mean
sample size. Run separate calls for different true medians. These are Monte Carlo
estimates, not exact OC or error-control guarantees.

This workflow assumes a constant event hazard and administrative censoring only.
It does not model dropout, competing risks, delayed data entry, or joint survival
and categorical outcomes. No calendar-time information after a stopping decision
is used to revise that decision.

Validation compares posterior probabilities with a 70-digit Erlang identity,
checks boundaries and both equality conventions, changes time units by factors
of `1e-200` and `1e200`, verifies censoring and absence of future-event leakage,
compares vectorized simulation with individual trial replay, and checks a
single-look simulation against its analytic success probability.
