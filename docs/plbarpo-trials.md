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

## Operating characteristics

`simulate_plbarpo` repeats the same controller with independent, recorded trial
seeds. It accepts the trial's design inputs and retains aggregate results and
seeds, discarding patient histories as each replicate finishes.

```python
from mdanderson_stats import simulate_plbarpo

simulation = simulate_plbarpo(
    [0.2, 0.7, 0.4],
    prior=np.ones((3, 2)),
    initial_active=[True, True, False],
    candidate_order=[2],
    min_n_per_arm=[1, 1, 1],
    max_n_per_arm=[2, 2, 2],
    max_total_n=6,
    look_sizes=[2, 4, 6],
    burn_in_per_arm=1,
    early_monitoring=False,
    theta_final=0.5,
    pfinal=0.8,
    null_arms=[True, False, True],
    trials=16,
    rng=2026,
)
efficacy = simulation.metric_index("any_efficacy")
print(simulation.metric_probability[efficacy])
print(simulation.familywise_false_efficacy_probability)
assert simulation.trials == 16
assert simulation.total_enrollment_mean == 6.0
```

Rows identified by `metric_names` report entry, futility, early efficacy, final
assessment, final efficacy, any efficacy and closure at the arm cap. Counts,
probabilities and binomial Monte Carlo standard errors use all simulated trials.
`any_efficacy` includes earlier declarations even when an arm has already closed.
`metric_given_entry` conditions on entry: its denominator is the corresponding
entry count, and an arm that never enters has `NaN` conditional results.

Mean arm assignments and responses, total enrollment and responses, no-efficacy
and early-stop rates are also returned. Mean standard errors use sample
variance; they are unavailable (`NaN`) for a single replicate. Probability
standard errors are the usual plug-in binomial estimates, including zero for an
observed rate of zero or one; they are not confidence bounds.

Error rates require `null_arms`, a full-ledger Boolean vector declaring which
hypotheses are null. Truth probabilities alone do not define the null. Without
that vector, false-efficacy and familywise-error fields are `None`. Familywise
error is the fraction of trials declaring at least one supplied null arm
efficacious, at either an early or final assessment.

To replay replicate `i`, call `run_plbarpo_trial` with the same design inputs
and `rng=int(simulation.trial_seeds[i])`. The result also records both underlying
stream seeds. Passing that seed to another `simulate_plbarpo` call starts a new
seed hierarchy, rather than replaying the original replicate.

Simulation is serial, capped at 5,000 trials and bounded by both per-trial and
aggregate work limits. These are numerical work proxies, not wall-clock
guarantees. The example's 16 trials demonstrate usage, not a precision study.

Control/concurrent-control platform scheduling, delayed responses, native
files and reports remain separate work. The existing
[control monitoring API](plbarpo-control.md) is available independently.
