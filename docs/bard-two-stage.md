# BARD stage-two continuation

`continue_bard_trial` takes a completed [BF-BLRM stage-one replay](bard-blrm-trials.md),
carries eligible patients at the two selected doses into the minimization
history, allocates new patients and applies the [final OBD rule](bard.md).
The paper's total target includes stage-one carryover. This completes a
configured two-stage path; accelerated titration, expansion, stage-two calendar
delays and hidden native settings remain separate scope.

```python
import numpy as np
from mdanderson_stats import BARDLogisticPrior, run_bard_blrm_trial, continue_bard_trial

arrivals = np.arange(0, 8.01, 0.5)
shape = (len(arrivals), 3)
stage_one = run_bard_blrm_trial(
    doses=[1, 2, 3], reference_dose=1,
    prior=BARDLogisticPrior(mean=[-3, 0], standard_deviation=[0, 0]),
    target_interval=[0.16, 0.6], eta=0.3,
    arrival_times=arrivals,
    potential_toxicities=np.zeros(shape, dtype=bool),
    potential_responses=np.ones(shape, dtype=bool),
    dlt_assessment_delays=np.full(shape, 2.0),
    response_assessment_delays=np.full(shape, 0.25),
    dlt_window=2, cohort_size=2, max_escalation_patients=8,
    backfill_evaluable_cap=3, draws=8, warmup=0, chains=2,
    rng=np.random.default_rng(165), boundary_policy="stop",
)
index = np.arange(len(stage_one.patients))
result = continue_bard_trial(
    stage_one,
    dose_pair=[1, 2],
    stage_one_eligible=index % 3 != 1,
    stage_one_factors=np.column_stack((index % 2 + 1, (index // 2) % 2 + 1)),
    candidate_factors=[[1, 1], [2, 2], [1, 2], [2, 1], [1, 1]],
    potential_toxicities=[[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
    potential_responses=[[1, 0], [1, 1], [0, 1], [0, 0], [1, 1]],
    total_target=10,
    prior=[0.25] * 4, safety_weights=[1, 1],
    allocation_probability=1, tie_probability=1,
    toxicity_limit=0.3, efficacy_limit=0.2,
    safety_cutoff=0.95, efficacy_cutoff=0.95,
    method="utility", utilities=[0, 30, 50, 100], margin=0.05, tie_arm=1,
    rng=np.random.default_rng(16503),
)
assert result.stage_one_carryover == 7
assert result.required_new_enrollment == result.stage_two_enrollment == 3
assert result.outcome_counts.sum() == 10
assert result.selected_arm == 1
```

The fixed prior, eligibility pattern and deterministic allocation probabilities
make this a bookkeeping example, not a calibrated design. Sampling precision
and stage-one model limitations are described in the linked guides. Stage two
uses the existing exact Dirichlet/Beta calculations and minimization rule.

The eligibility vector and historical factor rows align with
`stage_one.patients`, including both escalation and backfill patients. These
are enrolled-patient indices, not the original arrival indices. Only eligible
patients treated at either selected dose enter both balancing history and
final outcome counts. Excluded indices distinguish ineligibility from treatment
at another dose. Eligibility is supplied by the caller because the paper does
not prescribe clinical inclusion criteria.

`dose_pair` contains two ordered, one-based indices in the stage-one dose grid.
The paper selects these doses using the totality of the evidence and says the
higher dose is often the MTD; it does not require an adjacent pair. Arm 1 maps
to the lower supplied index and arm 2 to the higher. The continuation requires
a stage-one MTD recommendation and preserves permanent all-overdose stopping.
It returns `stage_one_no_mtd` without randomizing if that transition is unavailable.

New candidate rows are supplied in enrollment order and represent patients
already meeting stage-two eligibility. Potential outcome columns correspond
to the selected pair. Only the assigned column contributes observed outcomes.
Factor levels are positive integer categories, with one to five factors.
Each assignment retains its minimization scores, probabilities and random
seed; replay uses the same inputs and a freshly seeded generator.

For target `N2` and eligible carryover `n1`, exactly `N2-n1` new patients are
required. Extra candidate rows remain unused. A target below mandatory
carryover raises an error instead of dropping patients. If the tape ends too
soon, the result reports `candidate_tape_exhausted`, the shortfall, and no final
OBD. If carryover already meets the target, final selection uses it directly.
Noninferiority selection additionally needs observations in both arms; otherwise
the completed path reports `completed_selection_unavailable`.

Stage-two outcomes are assumed complete; the published simulation has no
interim stage-two toxicity/futility rule. The target is a combined total,
not a hard quota for each arm. The app's wording about patients per dose arm
does not resolve quota enforcement, so native quota parity is not claimed.
Prior shapes, safety weights, allocation/tie behavior and OBD settings are
explicit. [Existing source discrepancies](bard.md#explicit-choices-and-source-discrepancies)
also apply to final selection.

Combined factor-history and prospective minimization-work limits reject
oversized requests before randomization. The [independent audit](../research/bard-integrated-audit.md)
checks four full continuations against base-R ledgers, including eligibility,
selected-dose filtering, target completion, shortage and an already-full target.
