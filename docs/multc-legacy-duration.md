# Multc Lean legacy duration compatibility

`run_multc_legacy_duration` reproduces the recovered Multc Lean 2.1 duration
kernel for explicit compact stopping vectors and supplied random variates.
It records enrollment, outcomes, follow-up, balked arrivals, consumed draws and
stopping reasons. `multc_legacy_boundaries` now constructs those vectors from a
`MultcLeanDesign`, and `run_multc_legacy_design_duration` applies the mandatory
native prior screen. `simulate_multc_legacy_duration` adds bounded batch
simulation, native-defined study means and replayable Python seeds. Catalog
entry **12 remains partial**: native numerical-integrator/full-application
parity, saved native files/defaults and protocol/report workflow remain open.

The kernel was verified by running its original x86 instructions in an emulator,
with controlled external random draws and vector-access helpers. This verifies
the duration calculation, not the Windows application or its RNG. The original
DLL is a local research input and is excluded from distributions. See the
[audit](../research/multc-native-duration-audit.md) and
[source record](multc-legacy-duration-source.json).

## Replay a trial

```python
from mdanderson_stats import run_multc_legacy_duration

trial = run_multc_legacy_duration(
    6,
    # Earliest stopping size indexed by response count: 0 -> 3, 1 -> 5, ...
    response_stop_at=[3, 5, 7],
    nontoxicity_stop_at=[],  # disabled toxicity endpoint
    joint_probabilities=[0.25, 0.25, 0.25, 0.25],
    mean_interarrival=1,
    response_window=2,
    uniforms=[0.9] * 6,  # these draws select neither response nor toxicity
    unit_exponentials=[0.1] * 100,
)
assert trial.sample_size == 3
assert trial.balks == 19
assert trial.decision == "stop_response"
assert trial.duration == 2.2
```

`response_stop_at[r]` gives the earliest total treated count that stops with
`r` responses. `nontoxicity_stop_at[s]` is indexed by **NONtoxicities**, not
toxicities. A missing entry never stops; `max_subjects+1` also disables that
entry. Values must be integers 1 through `max_subjects+1`. A zero-enrollment
pretrial rejection must be handled before calling this routine. These vectors
are explicit inputs to this low-level routine. Use the design adapter below
for posterior/cohort conversion and automatic pretrial handling.

## Build boundaries and replay a design

```python
from mdanderson_stats import (
    multc_lean_design,
    multc_legacy_boundaries,
    run_multc_legacy_design_duration,
)

design = multc_lean_design(
    12,
    (1, 1),
    (1, 1),
    historical_response=0.5,
    historical_toxicity=0.5,
    response_cutoff=0.95,
    toxicity_cutoff=1,
    min_subjects=3,
    cohort_size=3,
)
bounds = multc_legacy_boundaries(design)
assert bounds.response_stop_at == (6, 9, 12, 12)
assert bounds.nontoxicity_stop_at == (12,)
trial = run_multc_legacy_design_duration(
    design,
    joint_probabilities=[0.25] * 4,
    mean_interarrival=1,
    response_window=2,
    uniforms=[0.9] * 12,
    unit_exponentials=[0.1] * 100,
)
assert trial.sample_size == 6
assert trial.decision == "stop_response"
```

The native prior-only comparison precedes the enrollment loop, regardless of
minimum enrollment. A rejected endpoint has vector `(0,)`. The adapter requires
`pretrial_check=True`; it rejects an incompatible design rather than silently
changing that setting. A rejected design returns zero enrollment, zero duration
and no consumed random variates, with decision `prior_response`, `prior_toxicity`
or `prior_both`. Supplied probabilities, times and unused arrays are still
validated.

At subsequent looks the original loop adds the cohort size and clamps to the
minimum enrollment. Its compact vector indexes successful outcomes; toxicity
is complemented to **nontoxicities**, swapping beta shapes, replacing fixed
history by `1-history` and negating the margin. Comparisons are strictly
greater than the cutoff. A probability equal to the cutoff continues.

The native vector ends with the enrollment cap even when the endpoint never
stops. This can be a **placeholder**, not adverse posterior evidence. In the
example, cutoff 1 disables toxicity but its vector is `(12,)`. The duration
kernel exits at the cap before testing actual stopping rules, preserving
`cap_complete`. Do not interpret these vectors as general clinical stopping
rules at or beyond the cap; the monitoring API exposes full posterior bounds.

## Run a replayable legacy batch

```python
from mdanderson_stats import simulate_multc_legacy_duration

simulation = simulate_multc_legacy_duration(
    design,
    joint_probabilities=[0.1, 0.1, 0.5, 0.3],
    mean_interarrival=0.2,
    response_window=2,
    trials=32,
    seed=2026,
    max_unit_exponentials_per_trial=1000,
)
print(simulation.summary.mean_duration, simulation.summary.duration_mcse)
print(simulation.summary.mean_sample_size)
assert simulation.summary.sample_size_probability.sum() == 1
replayed = simulation.replay_trial(0)  # zero-based replicate index
assert replayed.duration == simulation.summary.duration[0]
```

The native study means are duration, sample size, responses, toxicities and
balks, alongside the sample-size distribution. Python additionally reports
their Monte Carlo standard errors (sample SD divided by square root of trial
count), undefined for one replicate. Immutable replicate arrays include every
completed trial. Prior rejection has probability one at zero enrollment;
the native wrapper instead leaves that probability vector unpopulated.

Seeds use NumPy PCG64, with separate child streams for outcomes and unit
exponentials. They differ from the Windows generator. Simulation records all
child seeds and inputs and retains numeric summaries; individual patient
ledgers are reconstructed on demand. To aggregate externally replayed trials,
use `summarize_multc_legacy_durations(max_subjects, trials)`.

Batch limits are 10,000 trials, 1,000,000 exponential variates per trial,
50,000,000 generated draws by default and a conservative 64 MB default storage
budget. Each trial preallocates its configured exponential array. Work/storage
preflight precedes RNG creation; stream exhaustion raises an error without
returning a shortened trial or partial study. The draw budget charges the full
configured arrays, including unused variates. The method retains the historical
latent-count and balking behavior described below.

The four joint probabilities are ordered `[both, response only, toxicity only,
neither]`. Category cutoffs include equality, as in the binary. Uniforms must
lie in `(0,1)`. The exponential array supplies nonnegative draws from a unit-mean
exponential distribution, in the order they are consumed. Responders consume
one such draw for follow-up; arrivals and balks consume additional draws.
Extra valid inputs may remain unused. Stream exhaustion raises an error.

## Recovered timing and suspension behavior

The first patient enrolls at time zero. A responder's follow-up delay is
`min(response_window, unit_draw * response_window / log(20))`, using the
stored native constant `2.99573227355399`. This is **clipping**: there is 5%
probability mass at the window. Nonresponders use the entire window. The
legacy routine assigns one shared follow-up endpoint to the paired outcome;
it does not draw an independent toxicity time.

Before generating the next patient, it uses previous complete **latent**
outcome counts to determine whether that patient could cause a stop. When
that flag is true, later arrivals before the latest treated follow-up time
are discarded and counted as balks. The arrival clock continues to advance.
The actual stop is evaluated at the first remaining proposed arrival.
An arrival exactly at follow-up does not balk. At the enrollment cap the
routine exits before generating another arrival or evaluating stopping bounds.

The binary's duration is the latest treated follow-up time. A stopping decision
is evaluated at a later arrival, so `decision_time` can exceed `duration`.
Python exposes both times. It also exposes the original latent-count behavior;
this is distinct from the [calendar trial engine](multc.md), which bases decisions
on outcomes that have actually become observable and uses an explicitly
configured accrual clock. Use the legacy replay for historical simulation
compatibility, with that difference included in the study's methods.

The routine accepts 3–1000 maximum subjects and at most 1,000,000 supplied
exponentials. It rejects nonfinite times, overflow and positive increments that
cannot advance the floating-point clock. Its result owns an immutable patient
ledger; changing supplied inputs after completion does not change the result.

Thirty-four original-machine-code references cover disabled/single/dual
endpoints, clipping, all four joint categories, exact category cutoffs, balking,
varying window/accrual settings and early/cap exits. Focused tests additionally
check latent-count behavior, exact follow-up arrival, distinct clocks, stream
exhaustion and numerical/input limits. Random-stream or full-application parity
is not claimed.

An additional 22 design references execute the original compact-boundary and
toxicity-complement control instructions, using independently calculated base-R
posterior predicates. They cover 7,496 saved R probabilities, 54 composed native
duration replays and 22 original study-aggregation runs. The substituted
predicates do **not** establish native numerical-integrator parity; see the
[boundary/aggregation audit](../research/multc-native-boundaries-audit.md).
