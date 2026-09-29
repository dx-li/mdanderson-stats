# BF-BOIN accelerated titration

`simulate_bf_boin` supports the optional single-patient dose-escalation prelude
in the BF-BOIN guide's Remarks 2. It ends on the first DLT, second observed
grade-2 toxicity, or the configured cap rule. The ordinary BF-BOIN DLT safety,
cohort movement, and MTD selection rules are unchanged.

```python
from mdanderson_stats import BFBOINDesign, simulate_bf_boin

result = simulate_bf_boin(
    BFBOINDesign(n_cap=18),
    true_toxicity=[0.02, 0.05, 0.10, 0.18, 0.28, 0.40],
    true_response=[0.10, 0.20, 0.35, 0.50, 0.62, 0.70],
    cohorts=8,
    cohort_size=3,
    trials=20,
    accelerated_titration=True,
    titration_cap=6,
    true_grade2=[0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
    grade2_assessment_delay=1.0,
    rng=20260929,
)

print(result.titration_stop_reason)
print(result.titration_patients)
print(result.titration_end)
print(result.assigned)
```

`true_grade2` is a per-dose probability conditional on no DLT. This explicit
Python scenario makes the three severity outcomes (DLT, grade 2, and neither)
mutually exclusive. Grade-2 probabilities are sampled for all patients in an
accelerated run, including those enrolled after the prelude, so the returned
patient histories remain complete. `grade2_assessment_delay` is a fixed
positive time from enrollment. The source specifies the grade-2 stopping trigger but no grade-2
probability model or assessment-time distribution, so these choices are caller
inputs rather than recovered application defaults.

`titration_cap=None` uses the highest dose. Reaching the highest-dose cap ends
the singleton prelude upon enrollment; the simulator may start topping up the
first cohort at the next arrival before that singleton's outcome assessments.
With a lower cap, a trigger tops up the current dose, while reaching the cap
without a trigger starts a full cohort at the next dose. The top-up cohort
counts as the first of `cohorts`; earlier singleton visits are additional.
After the prelude exits, ordinary BF-BOIN cohort and backfill conduct applies.
Backfill does not occur during the singleton prelude.

The optional result fields expose per-patient grade-2 outcomes and assessment
times, titration exit reason/time, number of singleton visits, and grade-2
reports observed at the exit decision. The later patient histories can include
grade-2 assessments after the prelude has ended. `trial_duration` includes the
grade-2 follow-up horizon when accelerated titration is enabled.

This feature reuses the simulator's explicit Python arrival and DLT timing
conventions; it does not claim native application calendar or report parity.
The cached source is the BF-BOIN user guide's Remarks 2 (printed page 2;
cached PDF SHA-256 `65b01bbaf620d92ac03222a3dffb1f1f755808e5eb79e606d6356da4f0257e7d`).
See the [source audit](../research/bf-boin-accelerated-titration-audit.md) for
the exact conduct mapping and timing limitations.
