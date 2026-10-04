# Bayes Factor TTE scenario report

`bayes_factor_survival_report` creates a self-contained HTML snapshot from the
existing continuous boundary solver and calendar simulator. It records the
effective model and calendar inputs, scenario-specific child seeds, terminal
stopping probabilities, early and final-monitor probabilities, and patient
enrollment summaries. Optional boundary rows use continuous time-on-test roots
in the caller's unit.

```python
from mdanderson_stats.bayes_factor_survival_report import bayes_factor_survival_report

report = bayes_factor_survival_report(
    null_median=4.0,
    alternative_median_mode=5.5,
    inferiority_cutoff=0.15,
    superiority_cutoff=0.80,
    accrual_rate=2.0,
    max_patients=5,
    repetitions=3,
    check_times=[2.0, 4.0],
    final_followup=2.0,
    true_medians=[4.0, 5.5],
    seed=1234,
    time_unit="months",
    boundary_events=[0, 1, 2],
    max_boundary_rows=3,
    max_total_work=120,
    max_total_quadratures=36,
)
report.write_html("bf-tte-report.html")
```

The two true-median scenarios receive deterministic child seeds derived from
the top-level seed and are run serially. The report stores the actual child seed
for each scenario. For every scenario, true mean survival is reported as
`true_median / log(2)`. Terminal stopping uses the first early boundary crossed;
only if no interim stop occurred does the final-monitor result determine the
terminal stop classification. Early and final-monitor probabilities remain
separate because the fixed final follow-up can change evidence after an early
stop. Monte Carlo standard errors use the plug-in Bernoulli formula and are
undefined for a single repetition.

The aggregate worst-case work and Bayes-factor evaluation budgets are checked
across all scenarios before simulation. Their bounds use the existing kernel's
patient/check accounting. Boundary calculations are optional because each row
uses adaptive quadrature and repeated root evaluations. `max_boundary_rows`
limits the requested boundary rows before any boundary solve; it is a row cap,
not a claim that quadrature evaluations are exactly counted. The supported
event-count range is 0 through 500, and up to 501 rows can be requested by
setting the row cap accordingly.

All calendar assumptions are explicit Python conventions: first arrival at
time zero, exponential interarrival gaps at `accrual_rate`, exponential event
durations at each scenario median, caller-supplied absolute check times, and
final follow-up added after the later of the last planned arrival and final
check. The 2012 Guide lists accrual rate but does not specify native arrival
generation, check scheduling, or final-follow-up timing. Native calendar and
random-number parity are not claimed.

The Guide prints stopping boundaries in integer days. This report retains the
existing solver's continuous roots in the selected caller unit and does not
round or convert them to days. The native day discretization and root
approximation remain unresolved. This report is a static snapshot; reopening
the HTML displays the saved values and does not restore an editable session or
rerun calculations.
