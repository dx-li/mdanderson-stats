# TTEConduct static HTML reports

The TTEConduct guide describes output that echoes the seven design inputs and
shows a minimum total time-on-test boundary for each event count. It says the
output can be saved as HTML and a saved output reopened. The source does not
specify HTML markup or an editable project/session format, so this Python API
produces a self-contained static report. Reopening the file displays the saved
snapshot; it does not restore editable inputs or rerun the calculation.

```python
from mdanderson_stats import tteconduct_design, tteconduct_report

design = tteconduct_design(
    60, 295, 3, 10, 1, 0.03, 40,
    max_total_time=40 * 120,
)
report = tteconduct_report(design, time_unit="months", events=[1, 2, 3, 4, 5, 6])
report.write_html("tteconduct-boundaries.html")
```

The report factory validates the design and calculates the selected boundary
rows from that same design. It does not accept a separately supplied table that
could belong to different inputs. The seven source parameters are displayed
alongside Python's `max_total_time` search cap and `absolute_tolerance`. The
`time_unit` argument is a label only; every time quantity must already use the
same units. Python reports continuous roots at full float precision and does
not apply the native guide's month-to-day conversion or rounding convention.

Each boundary row includes the event count, minimum total time to continue,
posterior probability, quadrature error estimate, and probability residual.
A zero boundary means the continuation criterion is met with zero exposure for
that event count. A boundary that is not found within the configured cap is
shown as unresolved within that cap, with the probability and residual at the
cap; this does not claim that continuation is impossible. The report states the
strict `probability < cutoff` futility rule and separate maximum-patient stop.

Labels are escaped and bounded. Rendering finishes before the destination is
replaced; the completed UTF-8 file is published atomically. The HTML has no
external assets or scripts. It is a community report format, not byte-identical
native HTML and not an editable trial session.
