# CID2BP repeated comparisons and reports

`cid2bp_session` runs an ordered set of independent CID2BP comparisons and
methods through the existing `cid2bp_interval` calculation. A request can use
successes plus trial totals or successes plus failures, set its own confidence
level, and evaluate multiple methods in the requested order. The result keeps
the entered counts, normalized successes/failures/trials, requested method,
resolved method (so `auto` remains auditable), and interval.

```python
from mdanderson_stats import CID2BPRequest, cid2bp_session

session = cid2bp_session(
    [
        CID2BPRequest(7, 12, 1, 7, methods=("auto", "exact")),
        CID2BPRequest(
            10, 10, 0, 10, entry="failures", confidence=0.99,
            methods=("wald", "cox_snell"),
        ),
    ]
)
print(session.report())
session.write_report("cid2bp-session.tsv")
```

`entry="trials"` interprets each second count as total trials; `entry="failures"`
adds it to the successes. Each count is an integer, each resulting sample has
1–1,000,000 trials, and all nine methods retain their individual limits from
the interval API. A session is limited to 1,000 calculations. Requests must be
a sized sequence, and all inputs and aggregate limits are validated before
any confidence interval is calculated. Calculations run serially; an invalid
request fails the session instead of silently skipping a row.

The tab-separated report has one row per requested calculation, including the
sample counts and rates, confidence, estimate, requested and resolved methods,
and lower and upper limits. It defaults to eight significant digits and accepts
1–17 digits. `write_report` renders and validates the complete report before
opening the destination. This is a repeatable Python workflow; it does not
recreate CID2BP's prompt/retry loop or claim byte-identical native report
formatting.

The native program supports changing samples or confidence, applying several
methods to the current data, and recording the session in a report file. The
Python request sequence provides the same useful repeat-calculation pattern
without inventing a native batch-file format. The underlying statistical
methods and their limits are documented in [CID2BP](cid2bp.md).
