# BayesFactorTTE calendar replay and simulation

The existing [`bayes_factor_survival`](bayes-factor-survival.md) function
implements the exponential/iMOM Bayes factor and continuous time-on-test
boundaries. This extension applies that kernel to explicit calendars and runs
bounded serial operating-characteristic simulations. It does not claim
calendar or random-number parity with the 2012 native application: its guide
lists an accrual rate and reports stopping probabilities and patient-count
summaries, but does not specify the arrival law, interim-check schedule,
maximum-enrollment follow-up, or final analysis timing.

## Replay an explicit event tape

Supply one planned arrival and event duration per maximum patient, all interim
check times, and a prespecified final time. All times and median parameters
must use the same unit. A positive-infinite event duration represents no event
within the planned horizon; optional `censor_durations` are times from each
patient's enrollment and default to positive infinity.

```python
from mdanderson_stats.bayes_factor_survival_calendar import (
    bayes_factor_survival_trial,
)

trial = bayes_factor_survival_trial(
    enrollment_times=[0.0, 1.0, 2.5],
    event_durations=[1.5, 4.0, 2.0],
    check_times=[1.0, 2.0, 3.0],
    final_time=5.0,
    null_median=4.0,
    alternative_median_mode=5.5,
    censor_durations=[float("inf"), 3.0, float("inf")],
)

print(trial.stop_reason, trial.final_monitor.decision)
```

At a check, every patient arriving at or before that time enrolls first. An
event is observed when its absolute event time is at or before the check; an
event tied with its independent censor duration is counted as an event. A
censoring time tied with a check contributes its exact censor duration to
exposure. This tie convention is explicit for reproducibility; with continuous
event times these ties have probability zero under the simulation model.
Checks strictly increase; a check at `final_time` is reserved for final
monitoring. Checks before the first arrival do not evaluate the prior alone.

The first interim inferiority or superiority decision stops further
enrollment, and later arrivals/checks are ignored. `final_time` stays fixed
after that early stop, so enrolled patients can accrue additional follow-up.
The returned `early_monitor` and `final_monitor` are separate because added
follow-up can change the final decision. If no interim boundary is crossed,
the full arrival tape enrolls. The tape's last arrival and all checks must be
covered by `final_time`.

## Run a bounded simulation

The simulator uses an explicit Python timing policy: the first patient
arrives at time zero; later interarrival gaps are exponential with the
specified `accrual_rate`; `check_times` are absolute calendar times shared by
all trials; event durations are exponential with the supplied `true_median`.
The prespecified final time for each trial is
`max(last_planned_arrival, last_check) + final_followup`, even if an interim
decision stops accrual earlier. There is no additional random censoring in
this simulator; administrative follow-up ends at the final time. All time
inputs and the accrual-rate unit must be consistent.

```python
from mdanderson_stats.bayes_factor_survival_calendar import (
    simulate_bayes_factor_survival,
)

oc = simulate_bayes_factor_survival(
    null_median=4.0,
    alternative_median_mode=5.5,
    true_median=5.0,
    accrual_rate=2.0,
    max_patients=20,
    repetitions=100,
    check_times=[3.0, 6.0, 9.0],
    final_followup=3.0,
    seed=1234,
)

print(oc.early_superiority_probability, oc.final_superiority_probability)
print(oc.mean_patients_enrolled, oc.patient_count_quantiles)  # 10%, 50%, 90%
```

The summary reports interim and final inferiority/superiority probabilities,
plug-in binomial Monte Carlo standard errors (undefined for one replication),
mean patients enrolled and 10th/50th/90th patient-count quantiles. Returned
per-trial seed pairs provide exact replay independent of the parent seed:

```python
one_trial = simulate_bayes_factor_survival(
    null_median=4.0,
    alternative_median_mode=5.5,
    true_median=5.0,
    accrual_rate=2.0,
    max_patients=20,
    repetitions=1,
    check_times=[3.0, 6.0, 9.0],
    final_followup=3.0,
    trial_seed_pairs=oc.trial_seed_pairs[:1],
)
```

Accrual and event durations use separate random streams per trial. Simulation
is serial and does not retain trial histories. Preflight defaults cap work at
20,000 patient/check units and 1,000 Bayes-factor evaluations; callers may
raise these explicit limits up to 1,000,000 work units and 100,000 evaluations.
The retained results, current trial and conservative adaptive-quadrature
scratch must fit the requested storage budget, with a hard 128 MiB ceiling.
The deterministic replay accepts at most 500 patients and 1,000 interim
checks; aggregate repetition count is at most 20,000. These bounds keep large
quadrature workloads opt-in and serial.

## Source limits

The source guide documents the posterior cutoffs and native aggregate output
quantities, but not the timing rules needed to reproduce those quantities.
The calendar order and simulation distributions above are therefore explicit
Python conventions. The implementation does not claim native accrual,
monitoring, final-follow-up, report, or Monte Carlo parity. See the
[source audit](../research/bayes-factor-survival-calendar-audit.md) and the
base [posterior/boundary guide](bayes-factor-survival.md) for model details.
