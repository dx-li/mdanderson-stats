# BF-BOIN accelerated titration audit

The cached BF-BOIN user guide, `research/raw/BF-BOIN/Guide.txt`, Remarks 2
(printed page 2, lines 31–55 of the extracted text), describes an optional
single-patient-per-dose escalation before ordinary cohort dosing. The stated
triggers are the first DLT or the second moderate (grade-2) toxicity. At the
default highest-dose cap, reaching the highest level also ends titration and
the current level is topped up with `k - 1` patients. For a lower cap, a
trigger tops up the current level; reaching the cap without a trigger starts a
full cohort at the next level. The guide explicitly says not to backfill
patients during accelerated titration.

The corresponding cached PDF has SHA-256
`65b01bbaf620d92ac03222a3dffb1f1f755808e5eb79e606d6356da4f0257e7d`.

`simulate_bf_boin` implements those conduct branches while leaving the
existing BF-BOIN DLT safety, cohort movement, and MTD selection rules intact.
The `cohorts` argument remains the number of ordinary full cohorts. A trigger
or highest-cap singleton plus its top-up is the first such cohort; preceding
singleton visits are additional. A lower-cap transition without a trigger
starts the first ordinary cohort at the next dose. The possible singleton
staircase and its top-up reserve are included in request-size preflight.

Remarks 2 do not define a grade-2 probability model, grade-2 assessment time,
or calendar ordering. The simulator therefore requires callers to provide
`true_grade2` conditional on no DLT and a positive fixed
`grade2_assessment_delay`. The generated DLT, grade-2, and no-toxicity
categories are mutually exclusive; grade-2 outcomes are sampled for all
patients in an accelerated run so the returned histories are complete. A singleton below the highest cap advances
only after its DLT and grade-2 assessments are known; the highest-cap singleton
can be topped up at the next arrival before its assessments, as required by
the cap stopping rule. These are explicit Python timing conventions, not
claims of application calendar parity.

The optional result fields report per-patient grade-2 outcomes and assessment
times, the titration exit reason, singleton count, and number of grade-2
reports observed when the titration decision was made. The final patient
histories may contain grade-2 results observed after titration has ended.
`titration_end` records the calendar time the singleton prelude ends; the
trial duration includes the grade-2 follow-up horizon when this option is on.

## Validation

Eleven affected simulation/titration tests pass with warnings treated as
errors; targeted Ruff and mypy checks and the guide example pass. Every
existing result field reproduces 80 trials from published checkpoint
`d770fde`, spanning uniform/exponential arrivals and expansion on/off. The
worker's focused tests and baseline comparisons peaked at 128.93 MiB with
zero swaps.

An independent integration audit reconstructs 60 enabled-titration patient
ledgers from reported outcomes and observation times: 19 first-DLT exits,
16 second-grade-2 exits, 13 lower-cap transitions and 12 highest-cap exits.
It verifies serial singleton observation, absence of backfill during
titration, strict arrival chronology, grade-2 counts at exit, cohort-budget
accounting, exclusive toxicity categories and complete follow-up through
late grade-2 assessments. The combined audit with BOIN12's numerical
references took 0.265 seconds after imports, peaked at 125.12 MiB and
reported zero swaps. No native calendar random-stream parity is claimed.
