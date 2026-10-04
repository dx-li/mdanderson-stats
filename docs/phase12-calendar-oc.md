# Six-dose Phase I/II calendar operating characteristics

`simulate_phase12_calendar_oc` runs a bounded, serial collection of the six-dose
calendar trials and returns compact trial-level operating-characteristic
summaries. It retains one trial at a time; patient histories and posterior draws
are not accumulated across trials.

```python
from mdanderson_stats import simulate_phase12_calendar_oc

oc = simulate_phase12_calendar_oc(
    toxicity_probability=[0.03, 0.06, 0.10, 0.16, 0.24, 0.34],
    efficacy_probability=[0.12, 0.20, 0.31, 0.42, 0.48, 0.50],
    n_trials=20,
    seed=20260929,
    max_patients=30,
    max_attempts=250,
    draws=32,
    warmup=16,
    chains=2,
    optimal_doses=[2, 3],
)

print(oc.selection_probability)
print(oc.no_selection_probability, oc.mean_total_enrollment)
print(oc.generated_response_rate, oc.observed_response_rate)
print(oc.max_mcmc_split_rhat)
```

This short example demonstrates the workflow; 32 retained draws are not a
recommended calibration size for final operating-characteristic estimates.

The zero-based six-element selection arrays separate source early selections
from final selections. Their counts and probabilities use all completed trials;
`no_selection_*` is reported separately. Early selections remain in the raw
selection summaries even when the source marks the selected dose ineligible;
`early_ineligible_selection_*` makes this source behavior visible. The optional
`optimal_doses` result is only the probability that a selected dose belongs to
the caller-specified set. The implementation does not infer the optimal set or
define a separate success criterion.

Generated rates use the complete simulated response/toxicity truth for every
assigned patient. Observed rates instead use endpoint-specific denominators at
each trial's final analysis time, or its stop time if it stopped before a final
analysis. A pending response therefore does not enter the observed response
denominator, and a pending toxicity does not enter the observed toxicity
denominator. Rates pool records across trials by dose; their MCSE uses the
trial-level ratio influence contributions. Count and event-probability MCSEs
also use trials as the independent units. MCSE is undefined for one trial, and
a pooled rate is undefined for a dose with no observed endpoint records.

`mean_final_analysis_time_days` is conditional on
`final_analysis_trial_count`. `mean_enrollment_stop_time_days` summarizes the
last enrollment/stop time for every completed trial; it is not complete
follow-up duration. Setting `complete_followup=True` changes the
calendar trial's final follow-up behavior where applicable; it does not alter
the separate generated-truth totals.

The result also includes `duration_mean_months`,
`duration_population_variance_months_squared`, and
`duration_order_statistics_months`. These summarize each trial's enrollment
stop time, including early stops, after conversion from days using 12/365.
The seven entries use the archived C++ zero-based sorted indices
`n//40`, `n//20`, `n//4`, `n//2`, `n-n//4`, `n-n//20`, and `n-n//40`.
Their corresponding `duration_order_indices` make the convention inspectable;
an index at or above the number of trials is returned as NaN rather than being
clipped or replaced with an interpolated quantile. The C++ variable printed as
"Std" is actually calculated as the population variance
`E[D²] - E[D]²`; the Python field names it as variance. Existing duration mean
and MCSE remain in days. Source paths and the exact index/unit mapping are
recorded in the [duration audit](../research/phase12-calendar-duration-audit.md).

`phase_one_tally_*` fields reproduce the separate DF3Plus3 phase-I aggregate.
They include a trial only when the source calls `TallySim`: phase I has ended
and either proceeds to phase II, stops with at most one admissible dose, or
closes at the lowest dose for toxicity. The count and early-toxic-stop
probability use all simulated trials as the denominator. Per-dose patient,
toxicity, and admissibility means and MCSEs also use all trials; unfinished
phase-I paths contribute zero, matching the native accumulator. Toxicities
count generated DLT events, including outcomes not yet observed by the
calendar, because the native tally checks the event indicator without an
observation-time filter. `phase_one_admissibility_probability` remains the
separate probability of each dose being admissible at trial end, including
paths that do not reach the native tally point.
These phase-I means correspond to native output fields `#Patients`, `#Tox`,
`#Pat`, and `%Admissible` (where `#Tox` is a count); the reported MCSEs are
Python additions.

The optional `posterior_backend="importance"` uses the calendar's bounded
adaptive importance fit. Its summaries report fit counts, nonconvergence,
maximum ratio MCSE, raw component evaluations, and mode iterations. The
MCMC backend reports maximum split-R-hat and configured chain transition
slots. An infinite maximum diagnostic remains infinite; nonfinite fit counts
are reported separately. The MCMC transition count is a workload proxy, not a
count of likelihood evaluations. The elliptical-slice MCMC loop can make up to
1,000 shrink proposals for an update; this is distinct from the
importance backend's mode-iteration cap.

Trial `i` has the independent seed
`SeedSequence(seed, spawn_key=(i,)).generate_state(1, dtype=uint64)[0]`. Replay
that trial by calling `simulate_phase12_calendar` with the returned seed in
`default_rng`, the same scenario, timing, and backend settings. Posterior
randomness is kept on the calendar's separate child stream, so sampler draws do
not consume the data generator's stream. When decisions diverge, however, the
allocation and subsequent outcomes can also diverge.

Before creating trial seeds, the wrapper bounds requested trial count, worst-
case assignments and attempts, backend-specific posterior work, and a
conservative estimate of one active trial's allocations. These are explicit
Python resource policies, not a promise of total process peak memory; the
interpreter and numerical libraries have their own baseline allocations. The
default total work budgets are suitable for small checks; larger runs require
explicitly increased budgets within hard ceilings. No automatic parallelism is
used.

The archived C++ source also reports Laplace posterior-parameter summaries and
integrated posterior-probability summaries. Those require its final-fit
Hessian/mode and a source-specific probability-vector integration; the
existing Python calendar OC does not retain that same kernel output, so this
wrapper does not claim those summaries. See the
[source audit](../research/parallel-phase12-scenario-report-audit.md) for the
exact boundary.
