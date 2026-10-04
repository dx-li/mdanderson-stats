# BF-BOIN operating-characteristic report

`bf_boin_design_report` builds a static HTML report from the existing
`BFBOINDesign` and `simulate_bf_boin` APIs. It takes explicit toxicity and
response truth vectors for each scenario, captures the executed calendar and
cohort settings, runs scenarios serially from one seed, and retains summary
statistics rather than patient histories.

```python
from mdanderson_stats import (
    BFBOINDesign,
    BFBOINReportScenario,
    bf_boin_design_report,
)

report = bf_boin_design_report(
    BFBOINDesign(target=0.25, n_cap=12, n_stop=9),
    (
        BFBOINReportScenario(
            "Increasing toxicity and activity",
            true_toxicity=[0.05, 0.15, 0.30],
            true_response=[0.20, 0.40, 0.50],
        ),
        BFBOINReportScenario(
            "Nonmonotone toxicity stress test",
            true_toxicity=[0.20, 0.45, 0.25],
            true_response=[0.10, 0.30, 0.50],
        ),
    ),
    cohorts=6,
    cohort_size=3,
    trials=100,
    start_dose=1,
    accrual_rate=6.0,
    dlt_window=1.0,
    time_unit="month",
    arrival_distribution="uniform",
    seed=162,
)
report.write_html("bf-boin-report.html")
```

Scenarios are an ordered tuple of up to 20 unique labels and matching
per-dose toxicity and response vectors. Run parameters and all scenario inputs
are captured before simulation. A single NumPy Generator initialized from the
required integer `seed` is advanced serially across scenarios. Its random
stream is not intended to match the application or the independent CRAN
package. Each scenario obeys the simulator's limit of 1,000 retained patient
records per trial; the report preflights the complete planned record bound at
100,000 across all scenarios before starting any simulation.

The report contains dose truth, per-dose selection probabilities and MCSEs,
no-MTD probability and MCSE, mean assigned counts, mean completed counts,
mean DLTs, total assigned patients and DLTs, stop-reason frequencies, and mean
trial duration. It also records escalation-end timing and, when applicable,
post-escalation expansion and titration summaries. End-time means use only
trials with finite times, with their denominator shown. Grade-2 truth is
conditional on no DLT, as required by this Python simulator.

The cached Guide's Figure 15 labels a `% Pts treated` value. The report instead
labels its calculation `Allocated share`, defined as the mean assigned count
at a dose divided by total mean assigned counts over doses. The source does not
specify whether the native value is this ratio or the mean of within-trial
shares. The Figure 15 `% Early Stopping` value is likewise kept distinct from
the Python no-MTD selection probability and stop-reason frequencies; equality
is not assumed. The report does not claim native output-file or layout parity.

Timing and conduct details matter when interpreting the summaries. The first
arrival is at time zero; the arrival distribution is uniform or exponential.
DLT event times use the Python Weibull calibration `F(window)=p` and
`F(window/2)=p/2`, with assessment at the earlier of event time and window.
Responses are sampled at enrollment and observed at arrival plus the DLT
window. The report's time unit labels the units used consistently by accrual
rate, DLT window, and grade-2 assessment delay.

During ordinary escalation, the Python policy waits for the current cohort's
DLT assessments. Backfill uses observed response activity at the dose or below;
temporary closure requires both the dose-specific and adjacent pooled
evaluable DLT rates to exceed the de-escalation boundary. Pending outcomes are
not counted as non-DLTs, and empirical closure may reopen. `n_cap` limits
backfill assignments rather than ordinary escalation assignments. The
optional post-escalation expansion targets one level below the last dose
treated for escalation. These rules follow the audited primary method; the
calendar schedule is an explicit Python convention.

Current implementation differences remain visible in the report: the guide's
optional `stay_at_one_of_three=True` modification changes the individual action
at the current dose to stay for exactly one DLT among three at target 0.25.
When it applies to a current-dose action in a backfill conflict, Python applies
the modified individual action before the existing conflict-pooling rule; the
Guide does not state this interaction explicitly, so native parity for that
combination is unverified. BF extra-safe stopping uses the Guide's strict
`n > 3` requirement at dose 1, while ordinary dose elimination continues at
`n >= 3`. `bound_mtd=True` filters candidates whose fitted isotonic estimate
is not strictly below the de-escalation boundary. The CRAN package is an
independent implementation, not the application backend. See the
[source and reference audit](bf-boin-reference.md),
[calendar simulation guide](bf-boin-simulation.md), and
[titration guide](bf-boin-titration.md).
