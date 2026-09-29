# TITE-BOIN12 calendar replay

`run_tite_boin12_calendar_trial` follows a trial from staggered patient arrivals
through delayed toxicity and efficacy observations, enrollment suspension,
dose assignment and complete-outcome final selection. It supports both the
[approximate-likelihood](tite-boin12.md) (`method="al"`) and
[Bayesian data-augmentation](tite-boin12-bda.md) (`method="bda"`) decisions.
The calendar rules below are explicit Python choices. Exact native scheduling,
native random-number streams and full application parity are not claimed.

## Supply the patient and endpoint schedule

Use a consistent time unit for interarrival gaps, endpoint delays, assessment
windows and the decision lag. Each delay tape has shape `(patients, doses)`:
an entry is the time from enrollment to an endpoint event at that dose. Positive
infinity means no event during the assessment window. A finite delay must be
between zero and the corresponding window, inclusive. Only the entry for a
patient's assigned dose is observed by the conduct rule.

```python
import numpy as np
from mdanderson_stats import BOIN12Design, run_tite_boin12_calendar_trial

trial = run_tite_boin12_calendar_trial(
    BOIN12Design(0.35, 0.25),
    interarrival_gaps=[0, 1, 1, 1, 1, 1],
    toxicity_delay_tape=np.full((6, 2), np.inf),
    efficacy_delay_tape=np.full((6, 2), 1.0),
    toxicity_window=3.0,
    efficacy_window=4.0,
    cohort_size=3,
    decision_lag=0.5,
)
assert trial.enrollment_times.size == 6
assert np.all(trial.toxicity_outcomes == 0)
assert np.all(trial.efficacy_outcomes == 1)
assert trial.final_time >= trial.accrual_stop_time
print(trial.enrollment_times, trial.assigned_doses, trial.selected_obd)
```

The first cohort starts at `start_dose` (default 1) without an interim look.
The first gap sets the first patient's arrival from time zero. Within a cohort,
subsequent gaps are measured from the previous actual arrival, and every patient
receives the cohort's assigned dose. Decisions occur between complete cohorts.

At a cohort boundary, the next gap sets a candidate **analysis time** from the
last patient's actual arrival. If the current-dose pending fraction exceeds
either configured threshold, the analysis waits until the next pending endpoint
event or assessment deadline and tries again. An endpoint tied with an analysis
time is observed before the decision. Once enrollment is allowed, the first
patient of the next cohort arrives at that analysis time plus `decision_lag`.
Suspended patients do not accumulate in a queue. Thus a boundary gap and the
decision lag are separate time increments.

## BDA and final results

For `method="bda"`, supply `prior_concentrations` and an explicit
`numpy.random.Generator`. Shared four-cell priors or one four-cell prior per dose
are accepted, as in the BDA posterior API. Explicitly supply `draws`, `warmup` and
`chains` for the desired precision; the API does not infer missing native prior or sampler
defaults. Pending-fraction checks occur before sampling. For reproducibility,
recreate the generator from the same seed with the same input tapes and settings.

`steps` records each analysis, including suspensions, dose decisions, persistent
elimination and compact dose-level posterior summaries. Full BDA draw arrays are
released after each analysis. The result distinguishes the time enrollment stops
from the final analysis: enrolled patients are followed until both endpoints are
known, either through an event or the end of their window. Follow-up and outcomes
are returned along with dose-level counts and final OBD diagnostics.

A `stop_safety` enrollment decision forces `selected_obd=None`.
`selection` retains the separate complete-data selector diagnostics, even if
that diagnostic calculation would have selected a dose. Other stops and tape
exhaustion use the complete-data selector's OBD. Once all planned cohorts are
enrolled, the driver proceeds to final ascertainment; it does not add another
interim dose-allocation decision.

## Resource bounds and scope

Replay accepts at most 1,000 possible patients and 100 doses, with complete
cohorts and one gap per possible patient. Preflight checks bound calendar work,
compact history storage, per-analysis BDA work and total BDA work before sampling.
`max_calendar_work` and `bda_max_work` may lower the hard ceilings of 100 million
and 20 million work units respectively. These are computation estimates, not
elapsed-time promises or process-memory limits. Unrepresentable positive time
increments raise an error instead of silently changing chronology.

The [published patient ledger](../tests/fixtures/tite-boin12-paper-calendar.csv)
records the 18 patients in Table 2 of the
[TITE-BOIN12 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/).
Its [audit](../research/tite-boin12-paper-calendar-audit.md) distinguishes the
source observations from scheduling assumptions and unobserved potential outcomes.
It also records an unresolved day-315 dose-assignment discrepancy: the declared
AL rule de-escalates, while the paper illustration stays at dose 3. The example
therefore validates observation chronology, not full adaptive-trial parity.

This entry point replays supplied potential outcomes. It does not yet generate
correlated toxicity/efficacy scenarios or aggregate operating characteristics
over repeated trials. Categorical outcomes, native reports and the native
backend details listed in the AL and BDA guides remain outside its scope.
