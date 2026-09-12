# One Arm Time to Event Simulator

Catalog 98 provides single-arm exponential survival monitoring and simulation
against an uncertain historical standard. The implementation follows the
version 3.0.9 embedded help; see the [source review](one-arm-tte-source.md).

```python
from mdanderson_stats import one_arm_tte_design, simulate_one_arm_tte

design = one_arm_tte_design(
    standard_prior=[4, 12], experimental_prior=[1, 3],
    parameterization="mean", maximize=True,
    cutoff_inferiority=0.05, delta_inferiority=0,
    cutoff_superiority=0.95, delta_superiority=0,
    max_patients=40, minimum_patients=5,
    periodic_interval=1, monitor_at_accrual=True, followup_period=3,
)
result = simulate_one_arm_tte(
    design, true_tte=6, accrual_rate=2, repetitions=100, seed=98,
)
print(result.early_inferior_probability, result.final_superior_probability)
```

Each prior is an inverse-gamma **shape/scale pair on the chosen mean or median**.
`true_tte` uses that same parameterization. All times and margins use one
consistent time unit, and accrual rate is patients per that unit. The prior
elicitation APIs `solve_distribution_moments` and `solve_distribution_quantiles`
support the inverse-gamma family. A marginal improvement is additive; it is
not a hazard ratio. Setting `maximize=False` makes shorter event times better.
Each rule has its own margin and cutoff; setting a cutoff to `None` disables
that rule. At least one must remain enabled.

`one_arm_tte_monitor(design, patients, events, total_time)` computes the current
posterior comparisons. Total time is the sum of observed event times and
event-free follow-up, not calendar trial duration. Inferiority uses a strict
probability comparison below its cutoff; superiority uses a strict comparison
above its cutoff. Separate probabilities and flags preserve differing margins
and possible simultaneous rule triggers. The deterministic trial runner
enforces minimum enrollment and the monitoring schedule.

For a reproducible calendar analysis, call
`one_arm_tte_trial(design, enrollment_time, event_time)`. Supply exactly
`max_patients` potential arrivals in strictly increasing positive order and
positive event **durations from enrollment**. The runner stops adding patients
when a rule fires. Its returned patient records include only those actually
enrolled. `accrual_stop_time` and `final_time` are measured from trial origin
zero. `monitor_history` records accrual-phase checks; the final assessment is
returned separately as `final_monitor`. `early_monitor` is present only when
a rule stopped accrual early.
`event_calendar_time` gives each enrolled patient's supplied event date, which
may fall after the trial ends; an event is observed at the final assessment
only when that date is at or before `final_time`.

Periodic checks use the grid anchored at zero; pre-accrual checks evaluate
existing patients before the next enrollment. No check occurs before minimum
enrollment. Follow-up starts at the actual accrual stop, with no intervening
periodic checks, and ends in one final assessment. The reason for an early stop
and the final assessment can therefore differ. Events at a monitoring time
are included, and monitoring precedes an arrival at the same time. These exact
tie conventions are explicit Python behavior.

Simulation processes one trial at a time and retains compact per-replication
summaries: sample sizes, events, exposure, accrual-stop and final times, and
early/final rule flags. Inferiority and superiority frequencies are marginal
frequencies and can overlap when both rules fire. Sample-size and calendar-duration
quantiles summarize the simulated
trials; they are not posterior intervals for a model parameter. Monte Carlo
standard errors quantify simulation noise in the reported probabilities. A
zero estimated standard error does not establish that the true probability is
zero or one. Native random streams and sample-quantile interpolation are not
reproduced; the Python implementation uses NumPy and linear interpolation.
`credible_level`, default 0.95, selects the lower central quantile, median, and
upper central quantile. All returned arrays are read-only.

Designs support at most 1,000 patients. Each trial permits at most 10,000
accrual-phase monitoring checks. A simulation is bounded to 100,000 potential
patients and 100,000 accrual-phase checks across its repetitions; exceeding a
work limit raises an error rather than returning truncated trial results.
Zero margins use an analytic beta identity. Nonzero margins use checked
one-dimensional quadrature with estimated errors for each probability.
The requested `absolute_tolerance` is not silently relaxed when an integral
fails to converge, and it does not change the strict stopping comparisons.

The package provides numerical results rather than the desktop scenario editor,
interactive report history, or HTML save/reopen interface. The independent R
reference checks both goals, signed margins, and both parameterizations;
deterministic calendar examples check stopping, follow-up, and final
reclassification without relying on random simulations alone.
