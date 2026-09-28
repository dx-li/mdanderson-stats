# BARD BF-BLRM stage-one calendar replay

`run_bard_blrm_trial` combines the [paper's Bayesian logistic model](bard-blrm.md)
and [dose/backfill decisions](bard-blrm-decisions.md) with explicit patient
arrivals and assessment delays. It returns each assignment, posterior decision,
completed outcome and final MTD. [Stage-two continuation](bard-two-stage.md)
uses that result for eligible carryover, allocation and final OBD selection.
BARD remains partial: titration, expansion, stage-two timing and native reports remain open.

## A reproducible timeline

The example uses a point-mass prior to make every posterior decision exact and
illustrate scheduling. Its dose risks are fixed at `expit(-3 + dose)`; this is
an illustrative configuration, not an estimated or calibrated prior. For
posterior learning, supply positive prior standard deviations and select
sampling settings appropriate to the required precision.

```python
import numpy as np
from mdanderson_stats import BARDLogisticPrior, run_bard_blrm_trial

arrivals = np.arange(0, 8.01, 0.5)
shape = (len(arrivals), 3)
trial = run_bard_blrm_trial(
    doses=[1, 2, 3],
    reference_dose=1,
    prior=BARDLogisticPrior(mean=[-3, 0], standard_deviation=[0, 0]),
    target_interval=[0.16, 0.6],
    eta=0.3,
    arrival_times=arrivals,
    potential_toxicities=np.zeros(shape, dtype=bool),
    potential_responses=np.ones(shape, dtype=bool),
    dlt_assessment_delays=np.full(shape, 2.0),
    response_assessment_delays=np.full(shape, 0.25),
    dlt_window=2,
    cohort_size=2,
    max_escalation_patients=8,
    backfill_evaluable_cap=3,
    draws=8,
    warmup=0,
    chains=2,
    rng=np.random.default_rng(165),
    boundary_policy="stop",
)
assert trial.escalation_patients == 8 and trial.backfill_patients == 3
assert trial.accepted_arrival_indices == (0, 1, 5, 6, 7, 8, 9, 10, 11, 15, 16)
assert trial.final_assigned.tolist() == [5, 6, 0]
assert trial.selected_mtd == 2
assert trial.enrollment_stop_time == 8 and trial.final_time == 10
```

## Input and calendar contract

Arrival times must increase strictly. The four potential-outcome/delay arrays
have shape `(number of arrivals, number of doses)`. An assigned patient uses
the column for their selected dose. Declined arrivals retain their original
indices; no outcomes are recycled into later patients. Doses in results are
one-based, while arrival indices are zero-based.

DLT-positive assessments can occur before the DLT window ends. A negative
DLT assessment cannot occur before the full window. Response assessments have
their own delays, and both positive and negative responses become known at
their stated times. Delays must be finite and nonnegative; positive delays
that disappear when added to a large calendar origin are rejected.

All available assessments at a given time precede allocation at that time.
An escalation cohort has priority until it is full. The next escalation dose
is chosen after every patient in that cohort has a known DLT outcome;
lower-dose backfill can occur during that wait. Waiting for the full cohort's
DLT assessments is an explicit Python scheduling convention. The final cohort
may be partial when the escalation cap or arrival tape is exhausted.

Backfill eligibility uses observed responses and overdose probabilities. Its
cap counts **DLT-evaluable patients**. Pending assignments can therefore take
the eventual assigned count above the cap; imposing an assigned-patient cap
would implement a different rule. Escalation and backfill outcomes contribute
to the same toxicity likelihood. A fit is updated only when evaluable or DLT
counts change; a response-only event reuses the current posterior.

## Boundaries and follow-up

Initial prior screening, an unsafe current escalation dose, cutoff equality
with no safe candidate and an unsafe one-step move require an explicit policy.
`boundary_policy="raise"` raises an error; `"stop"` permanently closes
enrollment and precludes MTD selection. This is a Python policy for cases the
paper does not fully specify. Exact target-probability ties choose the lowest
dose, as in the decision helpers.

All assigned patients complete both assessments after enrollment stops.
An all-overdose finding is retained even if it first occurs during this final
follow-up or a later fit would recover. It prevents MTD selection without
rewriting the earlier enrollment-stop reason or time. `selected_mtd` is the
authoritative trial recommendation. `final_selection` exposes the raw
final-posterior candidate; it does not override an earlier permanent stop.

Final MTD eligibility requires a safe overdose probability and at least six
treated patients at that dose, including escalation and backfill patients.
`duration` uses arrival offsets and supplied delays to avoid subtracting large
absolute timestamps unnecessarily. `final_time` remains an ordinary floating
point calendar timestamp and can have coarser resolution at large origins.

## Diagnostics, limits and validation

Each event retains immutable counts, target/overdose probabilities,
log-parameter means, probability Monte Carlo errors, split R-hat and cumulative
work counters. Full posterior draws are discarded between fits. Inspect the
diagnostics when interpreting decisions near a cutoff; a completed replay
does not guarantee posterior precision or convergence. Constant quantities
can have undefined split R-hat.

Inputs are bounded to 2,000 arrivals, 100 doses, 200,000 cells per input matrix
and 1,000 escalation patients. Retained snapshot and per-fit arrays have
preflight bounds. Total likelihood and work budgets apply across all fits;
the replay raises clearly when a budget is exhausted. No internal parallel
workers are created.

Three independent fixed-prior R/hand-ledger examples cover pending backfill,
early DLTs, late responses, tied assessments and arrival-tape exhaustion.
All 16 patient records, as-of counts and fit-reuse counts agree. Independent
integration references validate the posterior model separately. See the
[calendar audit](../research/bard-blrm-calendar-audit.md) for exact scope and
memory results. No native timing distribution or native random-stream
equivalence is claimed.
