# ARAND simulation and operating characteristics

The [calendar controller](arand-calendar.md) can generate and summarize serial
trials with Poisson accrual, fixed-window binary outcomes or exponential event
times. It uses the same explicit controller policies and posterior calculations
as replay. Catalog entry 62 remains partial: native random streams, unspecified
controller conventions and desktop report equivalence are not established.

The version 5.2 guide describes these scenario families and per-arm selection,
patient-count and early-drop summaries on pages 9–10. This implementation
exposes separate arm-status events because the guide does not fully define how
its early-drop column combines temporary suspension and permanent elimination.

## Run and reproduce a binary scenario

```python
import numpy as np
from mdanderson_stats import (
    ArandControllerPolicy,
    ArandSimulationConfig,
    simulate_arand,
    simulate_arand_trial,
)

policy = ArandControllerPolicy(
    floor_transform="mixture",
    ranking_scope="all_arms",
    trigger_order=("futility", "suspension", "early_winner"),
    duration_minimum_precedence="duration_wins",
    multiple_winner="highest_probability",
    all_suspended_action="stop",
    same_time_order="analysis_before_arrival",
)
config = ArandSimulationConfig(
    prior=[[1, 1], [1, 1]],
    family="binary",
    analysis_times=[],
    max_enrollment=6,
    max_duration=None,
    final_followup=2,
    minimum_enrollment=2,
    policy=policy,
    binary_window=1,
    initial_equal_randomization=2,
    minimum_allocation=0.1,
    final_winner_cutoff=0.8,
)
result = simulate_arand(
    config,
    [0.8, 0.2],
    accrual_rate=2,
    max_candidate_arrivals=8,
    trials=4,
    seed=2026,
)
np.testing.assert_allclose(result.mean_patients_per_arm.sum(), 6)
np.testing.assert_allclose(
    result.selected_probability.sum() + result.no_winner_probability, 1
)
first = simulate_arand_trial(
    config,
    [0.8, 0.2],
    accrual_rate=2,
    max_candidate_arrivals=8,
    seed=int(result.trial_seeds[0]),
)
assert len(first.assignments) == 6
assert first.data_seed == int(result.trial_seeds[0])
```

This small example demonstrates the interface; use an adequate number of
replicates for a scientific operating-characteristic estimate. All configured
time quantities use the same unit. `accrual_rate` is the number of potential
arrivals per unit time. The first arrival and subsequent gaps are exponential;
no patient is forced to arrive at time zero. A duration cap can therefore close
a trial before anyone enrolls. Binary scenario values are per-arm outcome
probabilities; `binary_window` determines when each outcome becomes observable.

For exponential scenarios, set `family="exponential"`, omit `binary_window`
and supply positive true means or medians matching `parameter`. A median
scenario uses exponential sampling scale `median / log(2)`. Prior scales and
futility thresholds must also use the selected parameterization, as explained
in the [posterior guide](arand.md). Censoring follows the controller's analysis
and follow-up times. Separate child random streams generate arrivals, potential
outcomes and assignment uniforms. Reproduction requires the same configuration,
scenario, accrual rate and candidate cap.

## Interpret the summary

The returned immutable arrays include:

- Selection, early selection and final selection probabilities per arm, plus
  a no-winner probability. Every simulated trial contributes to the denominator,
  including zero-enrollment trials and trials without a winner.
- Probabilities an arm was ever temporarily suspended, permanently futile or
  displaced when another arm won early. These events can overlap and are not
  mutually exclusive reasons for being dropped.
- Mean patients per arm and empirical 2.5th–97.5th percentiles of those counts.
  Percentiles use linear interpolation (Hyndman–Fan type 7); they describe trial
  variability, not a confidence interval for the mean.
- Stop-reason frequencies and mean decision/completion durations. Duration
  meanings follow the calendar guide, including its early-stop convention.

Probability and mean estimates include Monte Carlo standard errors. A
single-replicate standard error is undefined and is returned as `NaN`. Count
percentiles remain defined for one trial. Trial seeds let callers reproduce
individual histories without retaining all histories in the summary.

## Resource limits and incomplete input

`max_candidate_arrivals` is an explicit work budget and must cover any
configured `max_enrollment`. Skipped arrivals while
all arms are suspended still consume candidates. If the budget prevents the
configured trial from completing, simulation raises an error instead of
silently treating a truncated trial as complete. A legitimate duration stop or
incomplete binary follow-up is retained under the configured trial policy.

The simulator checks worst-case trial work and aggregate work/storage budgets
before drawing seeds or candidate arrays. `max_total_work` and
`max_total_storage_bytes` may tighten aggregate limits; the controller's own
look and storage caps also apply. Trials run one at a time. Only seeds,
patient-count data needed for percentiles and compact summaries are retained.
