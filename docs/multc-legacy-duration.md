# Multc Lean legacy duration compatibility

`run_multc_legacy_duration` reproduces the recovered Multc Lean 2.1 duration
kernel for explicit compact stopping vectors and supplied random variates.
It records enrollment, outcomes, follow-up, balked arrivals, consumed draws and
stopping reasons. Catalog entry **12 remains partial**: native boundary-vector
construction, cohort/minimum-enrollment precedence and the full native study
workflow still need references.

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
are explicit inputs; no native posterior or cohort conversion is inferred.

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
