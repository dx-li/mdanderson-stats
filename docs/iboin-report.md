# iBOIN captured-input simulation reports

`simulate_iboin_report` connects the existing complete-outcome iBOIN simulator
to a self-contained HTML report and versioned JSON inputs. The report includes
historical skeletons, original and robust-effective ESS, true grade-2/DLT
probabilities, dose selection and no-selection probabilities, stopping
probabilities, enrollment and toxicity means, enrollment quantiles, and Monte
Carlo standard errors. All probabilities and count means use every simulated
trial as their denominator. A single repetition has undefined MCSE, explicitly
shown as unavailable.

Every assignment, stopping and final-selection setting is captured, including
custom isotonic weights, candidate-dose eligibility, prior-borrowing mode,
tie policy, final de-escalation-bound enforcement, starting dose, titration cap,
cohort size, patient budget and resource limits. Saved inputs contain exact
per-trial seeds, so replay does not depend on regenerating a root-seed sequence.
Use an integer-capable JSON reader: seeds can exceed the exact-integer range of
JavaScript numbers. Record the package source revision (`git rev-parse HEAD`)
separately; replay across different implementations is not guaranteed.

```python
from pathlib import Path
from mdanderson_stats import IBOINDesign, simulate_iboin_report, replay_iboin_report

design = IBOINDesign([0.10, 0.25, 0.40], [3, 3, 3], target=0.25, robust_prior=True)
report = simulate_iboin_report(
    design,
    grade2_probability=[0.05, 0.10, 0.10],
    dlt_probability=[0.02, 0.12, 0.30],
    cohort_size=3,
    max_patients=30,
    repetitions=100,
    prior_mode="effective",
    isotonic_weights="patients",
    seed=20261008,
    title="iBOIN scenario 1",
)
report.write_html("iboin-report.html")
report.write_inputs("iboin-inputs.json")
repeated = replay_iboin_report(Path("iboin-inputs.json").read_text())
assert repeated.inputs_json == report.inputs_json
```

`write_html` and `write_inputs` replace each destination atomically; parent
directories must already exist. The report needs no external scripts, styles,
network access or plotting dependencies. Titles and captured JSON are escaped
for HTML. JSON replay rejects unknown/missing fields, duplicate keys,
nonfinite constants, unsupported versions, inconsistent effective ESS and
invalid seeds. Input text is bounded to 4 MiB; the existing simulator enforces
its repetition, work and storage limits before trial generation.

This is a community Python reporting format, not native application report
parity. The [final-selection guide](iboin-final-simulation.md) explains the
explicit Python policies. Grade-2 and DLT outcomes are mutually exclusive
maximum-severity categories; assessments are complete, with no pending-event
calendar. The [main guide](iboin.md) records source-validated boundaries,
historical borrowing and conduct. Native isotonic weights, ties, final-prior
linkage and several final-selection conventions still need authoritative
numeric or source evidence. Catalog entry 145 therefore remains **partial**.
