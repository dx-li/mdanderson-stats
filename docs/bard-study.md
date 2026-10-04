# Saved BARD BF-BOIN scenario inputs

`BARDStudySpecification` captures the complete inputs to one repeated-trial
BARD BF-BOIN operating-characteristic scenario. It stores BF-BOIN settings,
toxicity truths, the response-model calibration inputs, stage-two allocation
and OBD settings, both caller-supplied true OBD dose labels, the simulation
settings, a distinct replay seed, and the patient-work budget. It reconstructs
the categorical response model through `bard_response_model` and delegates
every run to `simulate_bard_bf_boin`.

```python
from mdanderson_stats import (
    BARDStageTwoDesign,
    BARDStudySpecification,
    BFBOINDesign,
)

stage_two = BARDStageTwoDesign(
    total_target=14,
    eligible_profiles=[True, True, True, True],
    prior=[0.25, 0.25, 0.25, 0.25],
    safety_weights=[1, 1],
    toxicity_limit=0.30,
    efficacy_limit=0.20,
    safety_cutoff=0.95,
    efficacy_cutoff=0.95,
    utilities=[0, 30, 50, 100],
    margin=0.05,
    tie_arm=1,
    stage_two_accrual_rate=2.0,
    balanced_factors=(0, 1),
)
study = BARDStudySpecification(
    label="example",
    design=BFBOINDesign(target=0.25, n_cap=6, elimination_probability=0.99),
    true_toxicity=[0.08, 0.18, 0.30],
    population_response=[0.20, 0.32, 0.45],
    factor_profiles=[(1, 1), (1, 2), (2, 1), (2, 2)],
    profile_probabilities=[0.25] * 4,
    response_odds_ratios=[[1, 1.5], [1, 0.8]],
    stage_two=stage_two,
    true_obd_noninferiority=2,
    true_obd_utility=2,
    trials=20,
    seed=20261004,
    cohorts=3,
    cohort_size=3,
)
study.write_json("bard-example-inputs.json")
summary = BARDStudySpecification.read_json("bard-example-inputs.json").run()
print(summary.mean_total_enrollment.mean, summary.noninferiority_accuracy)
```

The specification snapshots array-like inputs as tuples and freezes the
stage-two arrays. `seed` is required and must be an integer in the unsigned
64-bit range; a mutable random generator is not accepted. This lets each
named scenario replay independently of execution order. `validate()` checks
the model, both outcome-joint distributions, complete OBD settings and resource
bound without running a simulation. `patient_work_bound` is the conservative
trial count times the maximum stage-one enrollment allowance plus inclusive
stage-two target; callers combining scenarios can sum it and preflight every
specification before starting any run.

Version-1 JSON is strict: unknown or duplicate fields, non-finite numbers,
unsupported versions, and inputs over 1 MiB are rejected. JSON contains the
response model's population margins, factor profiles and their joint weights,
and conditional odds ratios; fitted intercepts and repeated-trial output tables
are rebuilt rather than redundantly stored. `write_json` replaces a file
atomically. The portable Python schema is documented here and is not a claim
of byte-for-byte compatibility with native BARD save files.

The saved study retains the actual explicit inputs. The Python implementation
uses the current documented community API, so native GUI defaults and the
native Save Input / Save Scenarios file templates are not inferred. See the
[source and coverage audit](../research/bard-study-audit.md).
