# PLBARPO platform trials without a control

`run_plbarpo_trial` simulates a binary-response platform with a fixed number of
active slots and an explicit queue of replacement arms. Responses are complete
before the next patient. The entry, burn-in and transition rules below are
reproducible Python choices where the published guide does not fully specify
the native scheduler.

```python
import numpy as np
from mdanderson_stats import run_plbarpo_trial

trial = run_plbarpo_trial(
    [1.0, 0.5, 0.0], prior=np.ones((3, 2)),
    initial_active=[True, True, False], candidate_order=[2],
    min_n_per_arm=[1, 1, 1], max_n_per_arm=[2, 2, 2],
    max_total_n=6, look_sizes=[2, 4, 6], burn_in_per_arm=1,
    method="barn2n", theta_fut=0.5, pfut=0.99,
    theta_eff=0.5, peff=0.99, theta_final=0.5, pfinal=0.8,
    assignment_uniforms=[0.0, 0.9, 0.0, 0.9, 0.0, 0.9],
    outcome_uniforms=[0.2, 0.8, 0.2, 0.2, 0.2, 0.2], rng=2026,
)
np.testing.assert_array_equal(trial.assignments, [0, 1, 0, 2, 1, 2])
np.testing.assert_array_equal(trial.assigned, [2, 2, 2])
np.testing.assert_array_equal(trial.final_efficacy, [True, False, False])
assert trial.enrolled == 6
```

All arm inputs use the complete ledger, with zero-based indices in
`candidate_order`. The initial active set fixes the number of concurrent slots.
An arm enters at most once. Candidates fill vacant slots in the supplied order;
unused ledger arms stay unentered. Closed arms retain their data and contribute
to total enrollment in the BARN2N exponent, while receiving zero allocation.

Each entrant has its own burn-in quota. Randomization is uniform among active
arms still below their quotas, so existing adaptive arms temporarily wait while
newcomers complete burn-in. Adaptive allocation starts when the current burn-in
pool is empty. Thereafter probabilities stay fixed between global monitoring
looks, except when the active set changes. A replacement or cap closure forces
recomputation. Adaptive targets and floors apply outside burn-in. The trial
takes full-ledger DBCD targets, including future candidates, and normalizes
them over each active set. Floors likewise use full-ledger order and are
masked to zero for inactive arms before calling the
[active-arm allocation API](plbarpo-allocation.md).

`look_sizes` must increase strictly and end at `max_total_n`. At an interim
look, an arm must meet its own minimum enrollment before a futility or early
efficacy declaration. All decisions use the active set before any same-look
replacement. Contradictory simultaneous futility and efficacy are an error.
The [BARPO guide](barpo.md) describes the independent beta-tail criteria.

An arm closes as soon as its enrollment cap is reached, including between
scheduled looks. Its final efficacy rule is assessed at that cap. When a cap
coincides with a look, monitoring occurs before closure and replacement. The
global final look assesses eligible remaining arms and closes them at the
trial budget; no new arm enters then. If all arms close and no candidate
remains, the trial ends early. Cap closure, statistical stopping and the trial
budget have separate result labels. Earlier efficacy declarations remain in
`early_efficacy`; combine them with `final_efficacy` to count all declarations.

Results include immutable patient assignments, responses, allocation vectors,
look summaries and complete-ledger counts and decisions. Unassessed final
probabilities are `NaN`, with a corresponding `final_assessed` mask. Independent
assignment and outcome streams are derived from the supplied seed or generator;
the result records the two stream seeds. Optional uniform tapes are indexed by
global patient number and make assignments and outcomes directly replayable.

The trial API bounds the ledger at 100 arms, the active set at ten, enrollment
at 2,000 and monitoring at 200 looks. Its work limit also bounds repeated
posterior evaluations. These limits keep each trial compact and serial.

Three references from `tools/reference_plbarpo_trial.R` cover replacement after
a cap, consecutive futility/efficacy replacements, and all-arm futility without
another candidate. Independent beta tails and BARN2N weights match all 14
patient rows, 12 monitoring comparisons and nine final-ledger rows. The root
integrated check took 0.011 seconds after import, peaked at 117.69 MiB and reported
no swaps. Three focused regressions, targeted lint/type checks and the public
example also pass.
These establish the declared Python protocol, not native random-stream parity.

Control/concurrent-control platform scheduling, delayed responses, native
files and reports remain separate work. The existing
[control monitoring API](plbarpo-control.md) is available independently.
