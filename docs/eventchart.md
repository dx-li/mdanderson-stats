# EVENTCHART: event timelines

`event_convert` expands numeric time/event-code pairs into one time column per
observed event type. `event_chart_data` prepares calendar or elapsed-time event
charts, and `plot_event_chart` renders them using the optional `plot` extra.

```python
import numpy as np
from mdanderson_stats import event_convert, event_chart_data, plot_event_chart

converted = event_convert([[5, 1], [6, 0], [3, 1], [1, 1], [2, 0]])
times = np.column_stack((np.zeros(5), converted.times))
chart = event_chart_data(times, reference=0)
axes = plot_event_chart(chart, event_labels=("Entry", "Censored", "Death"))
axes.figure.savefig("event-chart.png")
```

![Calendar and elapsed-time event charts](eventchart-demo.png)

## Data and geometry

Inputs are numeric matrices with NaN for missing entries. Infinity and complex
values are rejected. Limits are one million rows, 1000 columns and two million
matrix entries; conversion also limits its expanded output to two million entries.
Negative times are allowed, including events before the reference time.
Numerical result arrays are immutable.

`event_convert(data, time_columns=(0,), code_columns=(1,))` uses zero-based column
indices. Supply equal-length index vectors for multiple time/code pairs. Codes
are sorted numerically within each pair; pair order is retained. NaN codes create
no event, and missing times remain NaN. With no observed codes, the result has
zero event columns. `names` optionally supplies names for all input columns;
output names combine the time-column name and code (e.g. `V1.0`, `V1.1`).
`source_columns` and `codes` retain the mapping to each output column.

`event_chart_data` accepts these layout options:

- `columns` and `rows`: nonempty, unique zero-based indices, defaulting to all.
- `reference`: original data column subtracted from every event column per row.
  `scale` is a positive divisor, such as 365 for elapsed years. Subtraction and
  division must produce finite coordinates wherever inputs are observed.
- `sort_by`: original data columns, in priority order. `ascending` is a boolean
  or one boolean per key. Sorting is stable; `na_last` explicitly controls missing
  keys in both directions. Sorting and `y_column` cannot be combined.
- `sort_after_subset=True`: first subset, then sort the selected records. When
  false, first sort all records and interpret `rows` as positions in that order.
- `drop_missing`: omit records with no remaining event times, including missing
  reference values. Otherwise they occupy empty rows. `renumber` places retained
  rows consecutively at 1,2,...; otherwise selected positions are retained.
- `y_column`: place records using a numeric covariate; missing covariates are
  omitted. `jitter` is a nonnegative factor multiplying the covariate range /
  `[2*(number of retained rows - 1)]`. A local generator controlled by `seed`
  supplies uniform offsets. Singleton and constant covariates have no jitter.
- `line_pairs`: pairs of selected original columns whose event times define
  extra intervals. At most 100 pairs are supported. Missing endpoints leave gaps.
- `all_rows_range`: use all original records to determine the x range, as in
  the source default; false uses only retained records.
- `labels`: one string per original record, carried through sorting/subsetting.

The result contains `times`, vertical `positions`, rowwise minimum/maximum
`spans`, extra `overlays`, original `rows` and `columns`, retained `labels`, the
`x_range`, and whether coordinates are `relative` to a reference. Missing-only
spans are NaN, not infinite pseudo-endpoints. A chart needs at least one retained
row and an observed event in its requested range.

The renderer returns its Axes without showing or saving. It accepts event labels,
marker shapes, span color and x-axis label; extra intervals are dashed. Numeric
calendar coordinates are accepted, but automatic calendar date formatting is
not yet provided. Use the returned Axes for styling and additional annotations.

## Native validation and deliberate corrections

The downloaded S `event.chart` routine was translated only from underscore to R
assignment syntax. Graphics callbacks captured its line and point coordinates.
The fixture compares calendar charts, reference-subtracted/scaled interval charts,
and multi-key sorted subsets with renumbering. The original `event.convert`
function ran unchanged for two time/code pairs with five output event columns.
Focused tests also cover missing references, aligned interval overlays, source-row
provenance, absent codes, deterministic jitter, overflow rejection and rendering.

Corrections to the original implementation:

- Subsetting before sorting can create missing y positions in the source because
  it indexes a shortened row-number vector with original row indices. Python
  retains the selected positions explicitly, including positions beyond the
  subset's length.
- After dropping missing references, the source may renumber even when renumbering
  is disabled. Python preserves the selected positions unless requested otherwise.
- Extra source intervals index unfiltered event rows after missing rows are
  removed. Python applies the same retained-row mapping to every geometry layer.
- Missing codes use an explicit mask, avoiding the source's NA replacement-index
  problem. Empty groups and singleton jitter do not invoke invalid ranges.
- Descending sorting keeps the requested missing-key placement. NumPy jitter is
  reproducible but does not reproduce S random numbers.

**Status: partial.** Remaining work includes Goldman/calendar-covariate charts
and their current-date boundary, calendar tick formatting and date conversion,
line styles grouped by covariates, fuller legend/style controls, and categorical
(non-numeric) coded-event inputs. Current calendar/interval geometry does not stand
in for those workflows.

Source: [MD Anderson EVENTCHART](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/32),
[original archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EVENTCHART/EVENTCHART_V1.tar.gz).
Lee JJ, Hess KR, Dubin JA, “Extensions and applications of event charts,”
The American Statistician 54:63–70 (2000). Source hashes: `eventchart-sources.json`.
