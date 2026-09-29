# BARD accelerated titration

`run_bard_blrm_trial(..., accelerated_titration=True)` adds the BARD guide's
one-patient-per-dose sequence to the [BF-BLRM calendar replay](bard-blrm-trials.md).
The guide describes BF-BOIN; this interface combines that sequence with the
paper's BF-BLRM model and the replay's explicit safety-boundary policy.

```python
import numpy as np
from mdanderson_stats import BARDLogisticPrior, run_bard_blrm_trial

arrivals = np.arange(0, 5.01, 0.5)
shape = (len(arrivals), 3)
trial = run_bard_blrm_trial(
    doses=[1, 2, 3],
    reference_dose=1,
    prior=BARDLogisticPrior([-3, 0], [0, 0]),
    target_interval=[0.16, 0.6],
    eta=0.3,
    arrival_times=arrivals,
    potential_toxicities=np.zeros(shape, dtype=bool),
    potential_responses=np.zeros(shape, dtype=bool),
    dlt_assessment_delays=np.full(shape, 2.0),
    response_assessment_delays=np.zeros(shape),
    dlt_window=2,
    cohort_size=2,
    max_escalation_patients=4,
    backfill_evaluable_cap=3,
    draws=8,
    warmup=0,
    chains=2,
    rng=np.random.default_rng(165),
    boundary_policy="stop",
    accelerated_titration=True,
    titration_cap=3,
    potential_grade2_toxicities=np.zeros(shape, dtype=bool),
    grade2_assessment_delays=np.full(shape, 2.0),
)
assert [(p.arrival_time, p.dose, p.role) for p in trial.patients] == [
    (0.0, 1, "titration"),
    (2.0, 2, "titration"),
    (4.0, 3, "titration"),
    (4.5, 3, "titration_topup"),
]
assert trial.titration_exit_reason == "highest_dose"
assert trial.titration_patients == 3
assert trial.escalation_patients == 4 and trial.backfill_patients == 0
```

The fixed prior makes this a scheduling example, not a calibrated design.
All grade-2 outcomes are negative here. The highest-dose top-up can begin at
time 4.5 even though the first patient's assessments at that dose occur at time
6. This differs from advancing to a higher dose, which waits for assessments.
The usual final MTD eligibility rules still apply; this four-patient example
does not provide six treated patients at any dose.

The grade-2 outcome and delay arrays have the same arrival-by-dose shape as the
other potential outcomes. Only the assigned column of a titration patient's
row contributes to the trigger count. Both arrays are required when titration
is enabled and rejected when it is disabled. Delays are finite and nonnegative;
they specify when the supplied grade-2 outcome becomes known. No grade-2 timing
distribution is assumed. Grade-2 events affect sequencing, while DLT outcomes
continue to determine the BF-BLRM likelihood.

Titration begins at dose 1. At most one titration patient awaits assessment.
Advancing to the next dose waits for both that patient's DLT and grade-2
assessments. Intervening arrivals are declined and recorded as
`titration_assessment_pending`; their outcomes are never recycled. A first DLT
or second observed grade-2 event among titration patients ends titration as soon
as observed and adds `cohort_size-1` patients at the current dose. Those top-up
patients have role `titration_topup` and complete the first regular cohort.

`titration_cap` is a one-based dose index and defaults to the highest dose.
Reaching that highest dose ends titration immediately and starts a same-dose
top-up. At a lower cap, absence of a toxicity trigger through both assessments
instead starts a full regular cohort at the next higher dose. No backfill occurs
while titration is active. Thereafter, the existing cohort and backfill rules
apply. Titration and top-up patients both count toward the escalation-patient
cap and remain available for [stage-two carryover](bard-two-stage.md).

The existing overdose-boundary policy applies to titration, top-up and ordinary
escalation assignments. It can preempt the titration sequence, as can permanent
all-overdose stopping. If highest-dose reach and the total escalation-patient
cap coincide, the patient cap takes reporting precedence and no top-up enrolls.
Final follow-up includes all assigned titration patients' grade-2 assessments,
even after enrollment ends; the final observed count need not equal the count
at the moment titration exited.

The serial assessment wait and BF-BLRM safety composition are explicit Python
conventions. Native BF-BOIN calendar equivalence is not claimed. The
[source and validation audit](../research/bard-accelerated-titration-audit.md)
records the guide's dose transitions and the focused calendar checks.
