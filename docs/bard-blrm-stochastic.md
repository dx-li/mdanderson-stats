# Stochastic two-stage BARD BF-BLRM trial

`run_bard_blrm_stochastic_trial` composes the generated BF-BLRM stage-one
calendar with the existing two-arm covariate minimization and both final OBD
analyses. The result retains the generated stage-one replay, stage-two
assignments, all factors, outcomes, calendar times, allocation decisions, and
the seed when it is reproducible.

```python
from itertools import product

from mdanderson_stats import (
    BARDLogisticPrior,
    BARDStageTwoDesign,
    BARDBLRMSimulationDesign,
    bard_response_model,
    run_bard_blrm_stochastic_trial,
)

profiles = list(product((1, 2), repeat=1))
response = bard_response_model(
    [0.30, 0.48, 0.62], profiles, [0.5, 0.5], [[1.0, 1.5]]
)
stage_two = BARDStageTwoDesign(
    total_target=12,
    eligible_profiles=[True, True],
    prior=[0.25, 0.25, 0.25, 0.25],
    safety_weights=[1.0, 1.0],
    toxicity_limit=0.30,
    efficacy_limit=0.20,
    safety_cutoff=0.95,
    efficacy_cutoff=0.95,
    utilities=[0.0, 30.0, 50.0, 100.0],
    margin=0.05,
    tie_arm=1,
    stage_two_accrual_rate=3.0,
    dose_pair=(1, 2),
    balanced_factors=(0,),
)
design = BARDBLRMSimulationDesign(
    doses=[1.0, 2.0, 3.0],
    reference_dose=1.0,
    prior=BARDLogisticPrior(mean=[-2.0, 1.0], standard_deviation=[0.0, 0.0]),
    target_interval=[0.15, 0.25],
    eta=0.40,
    cohort_size=3,
    max_escalation_patients=9,
    backfill_evaluable_cap=3,
    draws=8,
    warmup=0,
    chains=2,
    max_arrivals=40,
    accrual_rate=3.0,
    dlt_window=1.0,
)
trial = run_bard_blrm_stochastic_trial(
    design, [0.10, 0.22, 0.30], response, stage_two, rng=165
)
print(trial.status, trial.total_sample_size, trial.duration)
print(trial.selected_dose_noninferiority, trial.selected_dose_utility)
```

The example uses a point-mass toxicity prior only to keep a small workflow
demonstration inexpensive; it is not a paper-calibrated prior or an operating
characteristic estimate. For posterior uncertainty, provide nonzero prior
standard deviations and enough posterior draws for the intended precision.

The stage-two pair defaults to the selected MTD and its adjacent lower dose.
An explicit pair is allowed only after stage one has selected an MTD. Every
pair dose must satisfy the final strict BF-BLRM safety rule `POD < eta`; this
uses the final posterior and does not import BF-BOIN's persistent elimination
mask. A no-MTD result, missing default lower dose, or pair that fails the
final safety check is returned with an explicit status and no fabricated OBD.
An exhausted stage-one arrival schedule raises an error: the truncated replay
is not counted as a completed stochastic trial.

`eligible_profiles` controls both mandatory carryover and new stage-two
profiles. All stage-one patients on the selected pair with eligible profiles
are retained. The target includes that carryover; if carryover already exceeds
the target, no new patients are enrolled and the result reports
`carryover_exceeds_target`. The configured factor subset controls minimization;
the returned `factor_history` retains all factors so omitted-factor balance
can still be summarized.

Stage two starts after stage-one complete follow-up, places the first new
arrival at that time, and uses the configured uniform or exponential renewal
gaps. DLT and response assessments use the generated shared outcome-tape
policy. These are explicit Python timing and random-stream choices, not
recovered native timing or RNG parity. Without a joint endpoint probability,
DLT and response are generated conditionally independently given dose and
profile; a supplied joint truth changes the conditional response law. Both
final analyses use the same generated counts, so noninferiority calculation
does not consume another random stream.

The stage-one replay and its fit counters are available as
`trial.stage_one.trial`; generated tapes and captured seed are available on
`trial.stage_one`. The full patient-level and OBD result contract is described
in the [stochastic simulation audit](../research/bard-blrm-stochastic-audit.md).
