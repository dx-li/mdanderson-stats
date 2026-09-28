# PLBARPO platform trials with a persistent control

`run_plbarpo_control_trial` adds entire-trial or concurrent-control comparisons
to complete-outcome binary platform trials. Arm 0 is the control. Experimental
arms can stop, reach their enrollment caps and be replaced from an explicit
queue. The posterior criteria follow the PLBARPO guide; the entry and timing
rules below are declared Python conventions where the native scheduler is
not fully specified.

```python
import numpy as np
from mdanderson_stats import run_plbarpo_control_trial

design = dict(
    true_response=[0.5, 0.0, 1.0],
    prior=np.ones((3, 2)),
    initial_active=[True, True, False],
    candidate_order=[2],
    min_n_per_arm=[1, 1, 1],
    max_n_per_arm=[8, 2, 2],
    max_total_n=8,
    look_sizes=[2, 4, 6, 8],
    burn_in_per_arm=1,
    method="barcp",
    tau=0,
    early_monitoring=False,
    pfinal=0.85,
    assignment_uniforms=[0, 0.9, 0.9, 0, 0, 0.9, 0, 0],
    outcome_uniforms=[0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2],
    rng=2026,
)
entire = run_plbarpo_control_trial(**design, control_mode="entire")
concurrent = run_plbarpo_control_trial(**design, control_mode="concurrent")
np.testing.assert_array_equal(concurrent.assignments, [0, 1, 1, 2, 0, 2])
np.testing.assert_allclose(entire.final_efficacy_probability[2], 0.8)
np.testing.assert_allclose(concurrent.final_efficacy_probability[2], 0.9)
assert not entire.final_efficacy[2]
assert concurrent.final_efficacy[2]
```

In this example the replacement treatment enters after three patients. The
first control response predates its entry, so only the second control patient
is included in its concurrent comparison. The entire-trial comparison uses
both. This difference changes its final efficacy decision at the supplied
0.85 threshold; the paths and underlying control patients are otherwise
identical. `tau=0` gives equal adaptive weights outside burn-in here.

## Ledger and conduct

All arm inputs, including the beta `prior`, include the control at index 0.
The control must start active and cannot appear in `candidate_order`.
`max_n_per_arm[0]` must equal `max_total_n`; statistical rules never stop or
replace the control. At least one experimental arm must start active. The
initial active count fixes capacity, and candidates fill every vacant slot in
queue order. Arms enter once and never reopen.

Every entrant, including the initial control, has the supplied burn-in quota.
Randomization is uniform among currently under-quota arms. An older arm,
including the control, can temporarily wait while new arms complete burn-in.
Allocation floors apply only outside burn-in; they are not a guarantee of a
control allocation at every step. The four adaptive methods, full-ledger
targets/floors and global BARN2N enrollment follow the
[active allocation guide](plbarpo-allocation.md).

Allocation ranks the current active arms using their full observed histories.
Concurrent-control selection affects treatment-versus-control monitoring;
it does not replace the control in joint allocation with different synthetic
control arms. Allocation updates at global looks and active-set changes, with
immediate recalculation when entrant burn-in finishes. It stays fixed between
those updates.

Outcomes are available before the next assignment. Concurrent windows use
zero-based patient indices `[entry_index, stop_index)`: entry is the next
eligible patient's index, and closure is enrollment after the last eligible
assignment. For an open arm at a look, the right endpoint is current enrollment.
Replacement starts after the closing patient's observation. Entire mode uses
all control outcomes available at that assessment. Never-entered arms use `-1`
indices and an explicit `entered` mask.

An experimental arm can stop at a scheduled look once its minimum enrollment
is met. Futility requires `P(treatment <= control) > pfut`; early efficacy
requires `P(treatment > control) >= peff`. Either rule is optional. Conflicting
simultaneous declarations are rejected. An arm reaching its cap receives its
own final comparison using `pfinal`, without prematurely assessing other arms.
At a cap coinciding with a look, comparisons use the pre-transition active
set. Global final analysis assesses eligible remaining experimental arms.
The trial ends at its total enrollment limit or when no experimental arm and
no candidate remain. The control closes only with trial termination.

## Results and numerical scope

Immutable results retain patient assignments/outcomes, allocation histories,
arm counts, entry/stop indices, terminal reasons and separate early/final
efficacy flags. Combine those flags to count all efficacy declarations.
Look records retain the control counts, enrollment windows, comparison error
estimates and active sets before/after transitions. Final comparisons retain
their own control counts and error estimates; unassessed values are `NaN` and
identified by `final_assessed`. Control itself is never an efficacy hypothesis.

Assignment and outcome randomness use separate recorded streams. Optional
uniform tapes are indexed by global patient number, including entries that
may remain unused after early stopping. Bounds are 100 ledger arms, ten active
arms, 2,000 patients and 200 looks, with a shared numerical work limit. The
replay runs serially and reuses the validated posterior and allocation kernels.

Four independent base-R replays cover staggered replacement with both control
modes, a cap that must leave another experimental arm open, and simultaneous
futility with two replacements. The generator
`tools/reference_plbarpo_control_trial.R` records 94 allocation cells over 26
patients, 12 posterior comparisons and 14 terminal arm rows. These establish
the documented Python protocol and beta comparisons, without asserting native
scheduler or random-stream parity.

Delayed outcomes, control-trial aggregate operating characteristics and native
files/reports remain open. [No-control simulation](plbarpo-trials.md) is
available separately.
