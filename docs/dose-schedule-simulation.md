# Dose Schedule Finder operating characteristics

`simulate_dose_schedule_operating_characteristics` runs independent
[calendar trials](dose-schedule-trials.md) serially and summarizes selection,
stopping, patient allocation, toxicity and duration. Each trial uses the same
specified truth, prior, candidate schedules and arrival times. Full histories
and posterior draws are released between replicates.

```python
import numpy as np
from mdanderson_stats import (
    DoseSchedulePrior,
    simulate_dose_schedule_operating_characteristics,
)

prior = DoseSchedulePrior(
    mean=np.log([0.08, 2, 3, 0.08, 2, 3]),
    sd=[0, 0, 0, 0, 0, 0],
    dose_count=2,
)
oc = simulate_dose_schedule_operating_characteristics(
    truth_area=[0.08, 0.16], truth_peak=[2, 2], truth_tail=[3, 3],
    prior=prior, schedules=[[0], [0, 1]], horizon=6,
    arrival_times=[0, 2, 4], max_patients=3,
    toxicity_limit=0.5, upper_probability=0.8, target=0.2,
    trials=6, draws=8, warmup=0, chains=2,
    rng=np.random.default_rng(7501),
)
assert oc.selected_count.sum() + oc.no_selection_count == oc.trials
assert oc.stop_reason_count.sum() == oc.trials
np.testing.assert_allclose(oc.mean_allocation.sum(), oc.mean_enrollment)
assert np.all(oc.observed_toxicities <= oc.assigned_patients)
print(oc.selection_probability, oc.selection_mcse, oc.mean_duration)
```

The fixed prior and six replicates keep this example small. For an adaptive
posterior design, supply positive prior SDs and sampling settings suitable for
the model and data. Choose the number of independent trials for the desired
Monte Carlo precision; this example does not estimate a design's performance
accurately.

## Interpreting summaries

Dose/schedule arrays use zero-based dose rows and schedule columns. Selection
probabilities use all trials as their denominator; no selection is reported
separately. Stop counts distinguish the enrollment cap, no safe regimen and
no eligible regimen. A stopped trial completes follow-up but remains without
a final recommendation under the existing calendar policy.

Mean allocation, enrollment and duration have sample-mean Monte Carlo standard
errors. Selection and stopping probabilities use binomial plug-in errors.
At observed probabilities zero or one these errors are zero; that does not
establish that the true probability is exactly at the boundary.
Pooled toxicity in each cell divides total observed toxicities by total
assigned patients and uses a trial-clustered ratio standard error. Unvisited
cells have `NaN` rates/errors. With one replicate, mean and pooled-ratio errors
are undefined. Duration uses scaled online moments to avoid overflow from
squaring large but representable time units.

Final-fit parameter R-hat/Monte Carlo-error maxima and interim/final risk
diagnostic maxima accompany the summaries. Infinite diagnostics are retained;
maxima with no defined values are `None`. `diagnostic_undefined_values` counts
missing scalar summaries. These are diagnostic summaries, not convergence or
decision-accuracy guarantees.

## Replay and computational limits

`event_seeds[i]` and `sampler_seeds[i]` recreate replicate `i`: supply generators
initialized with their integer values as `rng` and `sampler_rng` to
`run_dose_schedule_trial`, with the same common configuration. This preserves
separate event and posterior streams. No new event-time law or allocation
rule is introduced by aggregation.

Up to 10,000 trials run serially. Output, one-trial history and fit storage are
checked together before advancing the supplied generator. Per-fit limits and
cumulative evaluation/work budgets bound the entire run. A failed replicate
raises its error; it is not dropped or replaced. Budgets limit computation,
not statistical precision or wall-clock time.

Three focused checks verify direct seed replay and count/error reconstruction,
budget rejection before RNG use, and duration invariance under time-unit
scaling by `1e200`. Existing R event and posterior references validate the
reused calendar/model calculations. Automatic calibration, delayed low-grade
toxicity classification, within-patient adaptation and native file/report
equivalence remain open.
