# Dose Schedule Finder calendar trials

`run_dose_schedule_trial` replays enrollment, treatment administrations,
toxicity observation and posterior allocation for the
[Dose Schedule Finder model](dose-schedule.md). Each patient is assigned one
dose and a nested administration schedule. The first receives the lowest pair;
later assignments use only information available at that arrival time.

```python
import numpy as np
from mdanderson_stats import DoseSchedulePrior, run_dose_schedule_trial

# Ordered areas use log positive increments; peak/tail times are fixed here.
prior = DoseSchedulePrior(
    mean=np.log([0.08, 2, 3, 0.08, 2, 3]),
    sd=[0.4, 0, 0, 0.4, 0, 0],
    dose_count=2,
)
trial = run_dose_schedule_trial(
    truth_area=[0.08, 0.16],
    truth_peak=[2, 2],
    truth_tail=[3, 3],
    prior=prior,
    schedules=[[0], [0, 1]],
    horizon=6,
    arrival_times=[0, 2, 4],
    max_patients=3,
    toxicity_limit=0.5,
    upper_probability=0.8,
    target=0.2,
    event_uniforms=[0.9, 0.05, 0.9],
    sampler_seeds=[751, 752, 753],
    draws=32,
    warmup=16,
    chains=2,
)
print(trial.final_decision.pair, trial.final_time)
assert len(trial.patients) == 3
assert trial.final_time == 10
assert trial.steps[1].decision is not None
```

This short run demonstrates the API and timing. Choose posterior sampling
settings for the desired precision; the example's small chains do not establish
accurate trial operating characteristics. `draws`, `warmup` and `chains` are
explicit inputs. Each interim step retains the risk/overdose estimates and
safety/allocation masks behind the decision, along with maximum risk split-Rhat
and Monte Carlo error diagnostics. The final full fit remains available for
further inspection. Diagnostics do not certify convergence or decision accuracy.

## Calendar and event model

All times use one consistent unit. Arrival times are absolute, nonnegative and
nondecreasing; their vector covers the configured maximum enrollment. Tied
arrivals are handled sequentially in input order. Relative administration times
must satisfy the existing nested-schedule validation.

For an assigned regimen, event generation inverts the sum of the triangular
cumulative hazards. It uses the inverse-CDF threshold `-log1p(-u)` for an event
uniform strictly between zero and one. A threshold beyond the hazard accumulated
by the horizon means no toxicity within follow-up. This preserves event timing
and overlapping administrations. A bounded root solve checks convergence and
its hazard residual; unresolved inversion raises an error.

At a new arrival, prior patients are censored at the elapsed time actually
available, capped by toxicity or the horizon. The fit includes only
administrations already received. Toxicity stops subsequent administrations;
it takes precedence at an exact administration-time tie. An event exactly at
the horizon is observed. No future toxicity or administration is used for an
interim allocation.

The existing posterior safety and no-skip rules choose subsequent regimens.
`no_safe_regimen` means none passes the safety screen; `no_eligible_regimen`
means the allocation constraints exclude the remaining safe options. Either
terminates enrollment permanently. Follow-up and the final fit continue, but
final recommendation remains `None` after such a termination. This explicit
Python policy avoids reopening a stopped trial; native post-stop recommendation
behavior is not established.

Final analysis occurs after every enrolled patient has an event or complete
horizon follow-up, and never before the decision that ended enrollment.
`stop_time` records enrollment termination, `final_time` records that final
analysis time, and `duration` starts at the first arrival. Duration is computed
from relative follow-up times to preserve precision when the calendar origin
is large.

## Reproducibility and limits

Event generation and posterior fitting use separate random streams. Supply
distinct NumPy generators as `rng` and `sampler_rng`, or explicit event-uniform
and posterior-seed tapes. The replay records full planned tapes, including any
unused suffix after early stopping, so they can be supplied directly to replay
the same design. Patient histories and compact interim decisions are immutable;
the full posterior draws are retained only for the final fit.

Limits include 200 patients, 20 doses, 20 schedules, the existing two-million-cell
bound on each retained fit, and shared evaluation/work budgets across all refits
and event generation. Static resource checks precede random draws. Numerical
limits bound work and memory, not wall-clock time or statistical precision.

Independent R quadrature and inversion provide 33 event references spanning
eleven scenarios and three time scales. A separate analytic rising-hazard
quantile at time `1e-50` checks an extreme small-probability case. These check
the event generator; existing posterior references validate the reused model.
They do not establish native executable or random-stream parity.

[Aggregate operating characteristics](dose-schedule-simulation.md) now run
these trials serially with replayable seeds and Monte Carlo error summaries.
[Explicit delayed-toxicity observations](dose-schedule-observation.md) provide
as-of event/censoring records for caller-supplied adjudication histories. They
are separate from this simulator's triangular-hazard event generator.
Automatic calibration, generated low-grade episode processes and native
files/reports remain open. The replay assigns
a fixed regimen to each patient; within-patient dose or schedule adaptation
requires a separately specified policy.
