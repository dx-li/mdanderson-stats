# Multc Lean and Multc99 Phase IIa

Catalog entry **12 (Multc Lean)** remains partial. Entry **3 (Multc99)** now
provides its [complete recovered general workflow](multc99.md). The existing
Python APIs in this guide provide marginal response/toxicity monitoring, full and reachable stopping
boundaries, sequential decisions, exact joint operating characteristics, and
calendar simulation with explicit endpoint timing.
This includes the fixed-reference Phase IIa rules described for Multc99;
Multc99's compound/conditional, mixture-prior, calibration and randomized
workflows are documented in its [general guide](multc99.md). The official
[Multc Lean catalog](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/12)
describes that relationship. [Source provenance](multc-sources.json) records the
versions and inspected documents. No original program code or binaries are
redistributed.

## Define a design

```python
from mdanderson_stats import multc_lean_design

# Published tutorial example; tuples contain beta success/failure shapes.
design = multc_lean_design(
    30,
    response_prior=(0.6, 1.4),
    toxicity_prior=(0.5, 1.5),
    historical_response=(30, 70),
    historical_toxicity=(20, 60),
    response_cutoff=0.95,
    toxicity_cutoff=0.95,
)
state = design.monitor(responses=0, toxicities=0, sample_size=6)
print(state.response_probability, state.decision)  # about .964594; stop_response
```

For each endpoint, observing `y` events in `n` patients updates the experimental
prior to `Beta(a+y,b+n-y)`. Historical beta distributions are unchanged and
independent of their corresponding experimental rates. A scalar historical
argument instead supplies a fixed rate. Response and toxicity in the **same
patient** need not be independent.

The [statistical tutorial](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcTutorial.pdf)
defines the response rule as
`Pr(historical_response + response_margin > experimental_response) > response_cutoff`.
The toxicity rule is
`Pr(historical_toxicity + toxicity_margin < experimental_toxicity) > toxicity_cutoff`.
Both comparisons are strict. A cutoff of one disables that endpoint. Fixed
thresholds outside the probability support are handled exactly, including a
zero cutoff with an impossible event.

`max_subjects` is 3–1000, the cohort size must divide it, and `min_subjects`
must be a cohort multiple or smaller than one cohort. Looks are the cohort
multiples at or above the minimum. Experimental beta shapes are in `(0,100]`,
historical shapes in `(0,1000]`, and margins in `(-1,1)`. Negative response and
positive toxicity margins cannot be combined, following the
[user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcUsersGuide.pdf).

`pretrial_check=True` applies the guide's prior-only rejection screen at zero
enrollment. Its interaction with minimum enrollment is not completely specified
in the guide. Python treats it as a separate screen; use `False` when minimum
enrollment must be guaranteed. This choice also applies to reachability and
operating characteristics.

## Monitoring and boundaries

`monitor(responses, toxicities, sample_size)` broadcasts independent snapshots
at scheduled looks or zero. It does not infer enrollment from the two event
counts, since events can overlap. `monitor_outcomes` instead consumes ordered
binary rows `(response,toxicity)` and evaluates their scheduled cumulative
looks. Its returned history ends at the first stopping decision, ignoring any
later supplied outcomes. Partial final cohorts are not evaluated.

`stopping_bounds()` returns `response_stop_max` and `toxicity_stop_min` at
each look: stop for response at counts at or below the former, or for toxicity
at counts at or above the latter. A response bound of `-1` or toxicity bound
of `n+1` means that endpoint never stops at that look.

`potential_boundaries()` gives compact inclusive count intervals attainable
after surviving earlier looks. An interval with lower endpoint greater than
upper endpoint is empty. Reachability allows all four paired-outcome categories;
a particular scenario with zero-probability categories may have fewer reachable
states. Because continuation restricts each count separately, surviving states
form a rectangle; the implementation propagates its two intervals without
retaining a large grid for every look.

At maximum enrollment, `cap_complete` takes precedence over posterior stopping
reasons. It means the sample cap was reached, not that the treatment passed an
efficacy or safety criterion. Raw endpoint bounds and probabilities remain
available at the cap. This also distinguishes them from the historical display's
final placeholders. The official
[boundary comparison note](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/Multc99vsMultcLean.pdf)
explains the differences between full and potential representations.

## Exact joint trial probabilities

```python
# Category order: both, response only, toxicity only, neither.
oc = design.operating_characteristics([0.12, 0.28, 0.18, 0.42])
print(oc.expected_sample_size, oc.expected_responses, oc.expected_toxicities)
print(oc.sample_size_probability)  # index n, including possible n=0

# Explicit independence assumption; equivalent to the above four probabilities.
independent = design.operating_characteristics_independent(0.4, 0.3)
```

A two-dimensional recursion advances joint response/toxicity count mass one
patient at a time, absorbing it at scheduled looks. Results distinguish response
only, toxicity only, both, and cap completion. They include the sample-size PMF,
mean and SD, and expected response/toxicity counts. Under bounded stopping and
iid patient pairs, Wald's identity gives each expected event count as its true
marginal rate times expected enrollment. Full path enumeration checks this
against direct event-count accumulation. Changing association while retaining
both marginal rates can change the stopping distribution.

## Multc99 Phase IIa mapping

Use scalar historical rates, zero margins and cohort size one. Multc99's
response rule compares `Pr(experimental_response > historical_response)` to
a **lower** efficacy cutoff. Thus its `.05` cutoff corresponds mathematically
to Python's `.95` response cutoff. The toxicity cutoff is unchanged. To monitor
one endpoint, set the other cutoff to one. Python computes CDF and survival
tails directly and does not reproduce native text placeholder conventions or
claim bit-for-bit parity at floating-point cutoff ties.

## Pending outcomes and calendar replay

`run_multc_calendar_trial` takes a design, a maximum-enrollment array of paired
binary outcomes, `max_subjects - 1` arrival intervals and separate response and
toxicity observation delays for each potential patient. It preserves the
specified association between outcomes and observation timing.

```python
import numpy as np
from mdanderson_stats import multc_lean_design, run_multc_calendar_trial

calendar_design = multc_lean_design(
    6,
    response_prior=(1, 1),
    toxicity_prior=(1.8, 0.2),
    historical_response=0.5,
    historical_toxicity=0.5,
    response_cutoff=1,  # disable response stopping
    toxicity_cutoff=0.8,
    min_subjects=2,
    cohort_size=2,
    pretrial_check=False,
)
trial = run_multc_calendar_trial(
    calendar_design,
    outcomes=np.zeros((6, 2), dtype=int),
    interarrival_intervals=np.ones(5),
    response_delays=np.zeros(6),
    toxicity_delays=[10, 9, 0, 0, 0, 0],
)
assert trial.arrival_times.tolist() == [0, 1, 11, 12, 13, 14]
assert trial.paused_duration == 9
assert trial.decision == "cap_complete"
assert trial.duration == 14
```

At a scheduled look, the replay considers every possible pending endpoint
count against the precomputed stopping boundaries. It continues immediately
if none can stop, stops if a stopping cause is certain, and otherwise pauses
until another endpoint becomes observable. Decision calculations never inspect
unavailable outcome values. Partial cohorts are not monitored. Events at the
current clock time are available, including zero-delay observations.

The first arrival is zero. Arrival intervals measure time while accrual is
open: a pause freezes this clock, and the next interval starts at resumption.
No arrivals are discarded or queued. This is an explicit Python convention;
the separately recovered legacy duration kernel discards arrivals as balks. Positive
intervals/delays that cannot advance the floating-point clock are rejected.

`stop_response` and `stop_toxicity` identify a guaranteed cause, not an exclusive
eventual reason: the other endpoint may also cross its boundary once pending
outcomes resolve. Look records expose observed event counts, pending counts,
actual thresholds and actions. `stop_both` means both causes are already
certain. `cap_complete` ends accrual without waiting for pending outcomes and
does not imply acceptable treatment performance.

The result distinguishes counts known at the decision from eventual totals.
`decision_time` and `accrual_end_time` mark enrollment termination;
`last_followup_time` is the last enrolled patient's endpoint availability;
`duration` includes all follow-up and cannot precede the decision. Returned
patient arrays cover only enrolled patients and are read-only. Work is bounded
at 12 million units; stopping boundaries are reused rather than reintegrated
at each calendar event.

The [user guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcUsersGuide.pdf)
specifies exponential arrivals and a truncated-exponential response-time
model. The recovered native duration kernel now establishes clipping and a
shared follow-up time, with discarded balked arrivals. See the separate
[legacy duration compatibility guide](multc-legacy-duration.md).
This calendar API retains its explicit timing inputs and observation-aware
policy. It does not claim native duration or random-stream parity. Five independent R examples
verify calendar paths using direct beta tails and enumeration of hypothetical
pending completions; see the [audit](../research/multc-calendar-audit.md).

## Simulate trial duration

`simulate_multc_trial` generates paired outcomes and observation times, then
runs the calendar replay. `simulate_multc` repeats this serially and reports
enrollment, eventual response/toxicity totals, stopping decisions, accrual and
follow-up duration, pause time, and Monte Carlo standard errors.

```python
from mdanderson_stats import MultcSimulationConfig, multc_lean_design, simulate_multc

duration_design = multc_lean_design(
    12,
    response_prior=(1, 1),
    toxicity_prior=(1, 1),
    historical_response=0.3,
    historical_toxicity=0.3,
    response_cutoff=0.95,
    toxicity_cutoff=0.95,
    min_subjects=3,
    cohort_size=3,
    pretrial_check=False,
)
timing = MultcSimulationConfig(
    duration_design,
    response_window=2,
    toxicity_delay=1,
    response_timing="conditional_truncated_exponential",
)
simulation = simulate_multc(
    timing,
    [0.12, 0.28, 0.18, 0.42],  # both, response only, toxicity only, neither
    accrual_rate=3,
    trials=32,
    seed=2026,
)
print(simulation.mean_duration, simulation.duration_mcse)
print(simulation.decisions, simulation.decision_probability)
```

The accrual rate and delays must use the same time unit. The first enrollment
is time zero; later gaps are exponential with mean `1/accrual_rate` on the
accrual-open clock. Durations therefore start at first enrollment. They do
not include waiting between study opening and the first patient. The
prior-only screen can instead stop a trial with zero enrollment and duration.

For responding patients, the underlying exponential response time has 95%
probability of occurring by `response_window`. The required `response_timing`
choice resolves the guide's ambiguous truncation convention:

- `conditional_truncated_exponential` conditions that time to fall within
  the window, with no point mass at its end.
- `clip_at_window` clips later response times to the window, putting 5% of
  responding patients at its end.

Nonresponses become known at the response window. Every patient's toxicity
status becomes known after the explicitly supplied `toxicity_delay`, including
those without toxicity. Joint outcome probabilities preserve response/toxicity
association; positive response times are generated independently of toxicity
status and accrual. For other timing models, supply observation times directly
to `run_multc_calendar_trial`.

The reported duration and accrual intervals are empirical 2.5th–97.5th
percentiles using linear interpolation, not confidence intervals for the mean.
MCSEs describe simulation uncertainty and are undefined (`NaN`) for a single
trial. All simulations, including prior-only rejections, contribute to the
denominators. `cap_complete` means maximum enrollment, not an efficacy or safety
declaration. Returned `trial_seeds` reproduce individual histories through
`simulate_multc_trial` with the same configuration, truth and accrual rate.

Up to 10,000 simulations run serially, releasing each history before the next
trial. Default preflight limits are 500 million work units and 512 million
estimated storage bytes; callers can lower them. These bounds cover retained
summaries and estimated live simulation storage, not the interpreter or
imported libraries. No native random-stream or duration-distribution parity
is claimed. See the [simulation audit](../research/multc-simulation-audit.md). The recovered
native timing policy is available separately through
[`run_multc_legacy_duration`](multc-legacy-duration.md), with explicit compact
boundaries and controlled random variates.

## Save a Python study input and scenario report

`MultcStudySpecification` captures the monitoring settings and a named list of
paired outcome-probability scenarios. Its exact operating characteristics use
the same `MultcLeanDesign.operating_characteristics` calculation above. An
optional `MultcStudySimulationSettings` runs the existing bounded calendar
simulation for each scenario; this requires an explicit response timing
convention and toxicity delay, as described above.

```python
from pathlib import Path
from mdanderson_stats import (
    MultcStudySimulationSettings,
    MultcStudySpecification,
)

study = MultcStudySpecification(
    12,
    (1, 1),
    (1, 1),
    0.3,
    0.25,
    [(0.12, 0.28, 0.18, 0.42), (0.05, 0.15, 0.35, 0.45)],
    scenario_names=("target", "high toxicity"),
    min_subjects=3,
    cohort_size=3,
    pretrial_check=False,
    simulation=MultcStudySimulationSettings(
        response_window=2,
        toxicity_delay=1,
        response_timing="conditional_truncated_exponential",
        accrual_rate=3,
        trials=32,
        seed=2026,
    ),
).run()
print(study.report())
study.write_specification("multc-input.json")
study.write_report("multc-scenarios.tsv", digits=17)
replayed = MultcStudySpecification.from_json(
    Path("multc-input.json").read_text(encoding="utf-8")
).run()
```

The report puts exact scenario operating characteristics apart from Monte Carlo
duration estimates, includes full and reachable stopping boundaries, and echoes
the settings. JSON inputs are data rather than executable code. Saved-input and
report layouts are community Python conventions; native Multc Lean configuration
and report files remain unsupported. The optional simulation uses the same
explicit Python calendar assumptions above and does not claim native timing or
random-stream parity. JSON uses a versioned schema and rejects inputs larger
than 1 MiB. Scenarios are limited to 100 per study; a 50-million default
aggregate exact-work cap and the existing per-call OC and simulation preflight
limits apply. This workflow covers Multc Lean's two-endpoint model only and does
not complete the broader Multc99 multiple-event program.

## Numerical scope and remaining work

Fixed-reference probabilities use direct beta CDF/survival functions. Random
references reuse the package's beta-difference quadrature, with refinement near
a cutoff and a clear error if the decision remains unresolved. Error estimates
include quadrature estimates rather than rigorous interval bounds. Arrays are
read-only; oversized broadcast requests are rejected before numerical copies.
Monitoring accepts at most 10,000 broadcast snapshots. Boundary construction
uses monotone searches with a 20,000-comparison budget. Exact operating
characteristics are limited to 10 million count-state updates and a 4 MB bound
on the two principal working arrays;
a valid design can therefore be too large for this bounded exact calculation.

Validation covers the published N=30 tutorial boundaries, Multc99's N=15
fixed-reference example, independent R integration of shifted beta tails, and
all `4**6` outcome sequences for small continuous/cohort designs. The reference
generator is [tools/reference_multc.R](../tools/reference_multc.R).
The nine focused checks passed in 1.64 seconds (1.73 seconds including test-runner
startup), with 133.2 MiB peak process RSS and no swaps. Ruff and the affected-module
type check also passed. The full repository test suite was not run for this
isolated addition; validation concentrated on the statistical calculations and
bounded allocations.

The native
[accrual logistics](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MultcLean/MultcLogistics.pdf)
allow accrual to continue while outcomes are pending when they cannot change the
next decision; the replay and simulation above implement that decision logic. Native
configuration/report formats, protocol documents, and
the general Multc99 multiple-event workflow are also pending. Parameter
elicitation and distribution inequalities already have separate package APIs:
`solve_distribution_moments`, `solve_distribution_quantiles`, and
`compare_beta_difference`.
