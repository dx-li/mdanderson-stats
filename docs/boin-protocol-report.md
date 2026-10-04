# BOIN design and operating-characteristic report

`boin_design_report` combines the existing `BOINDesign`, `boin_protocol`, and
`simulate_boin` interfaces into a bounded, reproducible report. It calculates
the protocol text and simulation summaries from the same captured design and
settings. The saved HTML is a static snapshot; it does not rerun simulations or
contain trial-level count arrays.

```python
from mdanderson_stats import BOINDesign, BOINReportScenario, boin_design_report

design = BOINDesign(target=0.30, extra_safe=True)
report = boin_design_report(
    design,
    (
        BOINReportScenario("Increasing toxicity", [0.05, 0.15, 0.30, 0.45]),
        BOINReportScenario("Nonmonotone stress test", [0.10, 0.35, 0.30, 0.55]),
    ),
    cohorts=10,
    cohort_size=3,
    trials=1000,
    seed=20261004,
)
report.write_html("boin-design.html")
```

Scenarios are an ordered tuple of unique labels and one true DLT probability
per dose. Reports accept at most 20 scenarios, 1,000,000 trial-dose cells per
scenario, and 2,000,000 planned patient-replications total. The plan may enroll
at most 200 patients. Simulation proceeds serially from one explicit NumPy
seed; streams do not match the native R implementation. The report retains
selection probabilities and MCSEs, mean enrollment and DLT counts by dose, stop
reason frequencies, titration summaries, and source-defined allocation-risk
summaries. No MTD is an explicit selection category.

When accelerated titration is enabled, `moderate_toxicity` gives the per-dose
grade-2 probability, mutually exclusive with DLT. The vector is checked against
every scenario before simulation. An omitted cap is recorded as the highest
dose. The report shows grade-2 counts only for the titration phase, matching
`simulate_boin`.

The optional overdose-risk section follows the cached BOIN and Keyboard
operating-characteristic sources: a dose is above target when its true DLT
probability is strictly greater than the target, and the event is more than
60% or 80% of planned enrollment at those doses. The report displays Python
fractional probabilities and Bernoulli MCSEs; the native application prints
percentages. These estimates are omitted with an explicit reason unless at
least one truth probability equals target. Source provenance and edge-case
notes are in [the allocation-risk audit](../research/dose-allocation-risk-audit.md).

The HTML safely escapes scenario labels and the generated English or Chinese
protocol text. It has no animation or live decision interface, and it does not
add observed-patient decisions to a design report. See [the BOIN guide](boin.md)
for the decision, safety, estimation, and simulation contracts.
