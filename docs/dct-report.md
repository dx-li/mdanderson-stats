# DCT sample-size report

`dct_sample_size_report` runs one existing DCT sample-size planner and returns
an immutable, readable report of the effective inputs and allocation. Set the
endpoint to `"continuous"` for the weighted mean-difference z-test or `"binary"`
for the weighted difference-in-proportions normal approximation.

```python
from mdanderson_stats import dct_sample_size_report

report = dct_sample_size_report(
    "continuous",
    allocation_unit="participants",
    effect=10,
    onsite_sd=20,
    offsite_sd=25,
    relative_bias=-0.2,
    offsite_fraction=0.75,
    power=0.8,
    alpha=0.05,
)
print(report.report())
report.write_report("dct-sample-size.txt")
```

The allocation unit is required. Choose `"participants"` when the repeat
settings describe repeated measurements on participants, or `"clusters"`
when they describe cluster sizes. The report presents the planner's allocation
in that chosen unit and never converts cluster counts into participant counts.
The returned report records all planner defaults, the unrounded total, each
rounded stratum-arm count, the rounded total, achieved and target power,
significance level, test sides, and citation. Each active allocation cell is
rounded upward independently; no post-hoc adjustment is made to match an app
example. Omitted experimental SDs for a continuous endpoint are recorded as
their effective values (the corresponding control SD). The planner does not
include dropout inflation.

The DCT page's result panel is dynamically rendered, so its exact native
caption, table, and citation layout is not reproduced. For the fully offsite
continuous help example (effect 10 and SD 20), the app help lists 128 total
participants while the implemented formula and independent upward rounding
give 126. The report preserves the Python result and flags that known difference
for this exact input case. Continuous and binary methods use a weighted z-test
approximation; the report is not an exact finite-sample test or dropout-adjusted
design.

Source details and the result-field mapping are in
[`dct-report-audit.md`](../research/dct-report-audit.md); source hashes are in
[`dct-sources.json`](dct-sources.json).
