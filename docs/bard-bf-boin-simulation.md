# BARD BF-BOIN repeated-trial operating characteristics

`simulate_bard_bf_boin` runs serial BF-BOIN-to-BARD trials and returns online
operating-characteristic summaries without retaining every patient ledger.
The source paper reports total sample size, duration, dose-arm allocation
imbalance, per-factor imbalance, and correct OBD selection for both final
selection methods.

The API takes the same stage-one truth, response model, and stage-two design as
`run_bard_bf_boin_trial`, plus the number of repetitions and the two
caller-supplied one-based true OBD labels:

```python
from mdanderson_stats import (
    BARDStageTwoDesign,
    BFBOINDesign,
    bard_response_model,
    simulate_bard_bf_boin,
)

response_model = bard_response_model(
    [0.20, 0.32, 0.45],
    [(1, 1), (1, 2), (2, 1), (2, 2)],
    [0.25, 0.25, 0.25, 0.25],
    [[1.0, 1.5], [1.0, 0.8]],
)
stage_two = BARDStageTwoDesign(
    total_target=14,
    eligible_profiles=[True, True, True, True],
    prior=[0.25, 0.25, 0.25, 0.25],
    safety_weights=[1.0, 1.0],
    toxicity_limit=0.30,
    efficacy_limit=0.20,
    safety_cutoff=0.99,
    efficacy_cutoff=0.99,
    utilities=[0.0, 30.0, 50.0, 100.0],
    margin=0.05,
    tie_arm=1,
    stage_two_accrual_rate=2.0,
    balanced_factors=(0, 1),
)

summary = simulate_bard_bf_boin(
    BFBOINDesign(target=0.25, n_cap=6, elimination_probability=0.99),
    [0.08, 0.18, 0.30],
    response_model,
    stage_two,
    true_obd_noninferiority=2,
    true_obd_utility=2,
    trials=100,
    cohorts=3,
    cohort_size=3,
    rng=20261004,
)
```

The example is a small workflow demonstration, not a calibrated study. In a
real analysis, set the two true OBD labels from the protocol's scenario record;
the simulator does not infer them. Construct `response_model` from an explicit
factor-profile distribution and response truths. Construct `stage_two` from
the protocol's eligibility, dose pair or default pair rule, target, prior,
cutoffs, and allocation settings. The scenario's response and toxicity marginal truths do not identify
a joint toxicity-response distribution; pass `joint_toxicity_response_probability`
to state one explicitly. Omitting it uses conditional independence in the
Python simulator. Stage-two profile weights are conditioned on the same
eligibility mask used for stage-one carryover.

The selection-by-dose probabilities, no-selection probabilities, and
unconditional correct-selection probabilities use all trials. The accuracy
conditional on a method selecting a dose is returned separately with its own
selected-trial denominator. Pair allocation imbalance and factor imbalance
have distinct denominators: arm-count difference is defined for every
non-rejected pair, including an empty arm; factor level-1 proportion difference
is defined only when both combined eligible arms are nonempty. Factor
differences are proportions on the 0–1 scale, not percentages. No-pair,
safety-rejected-pair, and empty-arm trials remain explicitly counted rather
than becoming zero-valued imbalance observations.

One NumPy Generator is consumed serially. Passing an integer (or `None`)
initializes the stream; passing a `Generator` advances it in place. The
`max_patient_work` guard bounds the worst-case sum of stage-one enrollment
allowance and stage-two target across all trials. It is an upper bound on
patients simulated, not a processor-time estimate.

The [source audit](../research/bard-bf-boin-simulation-audit.md) separates
paper-defined operating characteristics from timing, joint-outcome, and
native-interface conventions that remain unspecified.
