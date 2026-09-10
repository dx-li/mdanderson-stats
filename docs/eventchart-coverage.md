# EVENTCHART coverage audit

The public archive's `event.chart.d`, `event.convert.d`, implementations and
worked examples define two entry points and three principal plot families.
This audit maps those calculations and plotting workflows to Python. The
catalog's **implemented** status covers these families, not binary or calling-
convention compatibility with an S graphics device.

| Source capability | Python implementation and evidence |
| --- | --- |
| Coded time/event conversion, multiple pairs, factor levels | `event_convert`; native numeric and factor/string fixtures, including an unused level and shared time column |
| Calendar event chart | `event_chart_data`, `plot_event_chart(calendar=True)`; native event/span coordinates and scaled calendar rendering |
| Interval/aligned event chart | Per-row `reference` and positive `scale`; native aligned coordinates, before-reference times allowed |
| Goldman chart and current-date line | `goldman_chart_data`, `plot_goldman_chart`; native full/subset boundary fixtures and corrected calendar identity |
| General numeric/date covariate y axis | `y_column`, optional local jitter, `calendar_y`; missing covariates removed with provenance retained |
| Source boundary with a distinct calendar covariate | `goldman_chart_data(y_column=..., native_boundary=True)`; original intercept/slope fixture |
| Sorting, multiple keys, subset before/after sorting | Stable `sort_by`, directions and missing-key placement; native sorted-subset geometry and original-row mapping checks |
| Missing-only rows, reference omissions and renumbering | Explicit `drop_missing`, `renumber`, immutable original row indices; focused missing-reference/overlay regression |
| Grouped subject lines | `line_groups`, `EventLineStyle`, `group_styles`; native coordinate, line-type and width fixture after sorting/subsetting |
| Extra intervals | `line_pairs`, `overlay_styles`; retained-row alignment check, per-pair appearance |
| Per-event symbols, sizes and colors | `markers`, `point_sizes`, `point_colors`; renderer accepts one value per selected event, including repeated symbols |
| Calendar origins and date labels | `event_dates`, `event_date_labels`, explicit origins, ISO labels; leap-year and scaled-axis checks |
| Square Goldman plot | `square=True`: square physical plotting box, current-date extent and boundary; rendered example |
| Internal legend and explicit legend placement | `legend`, `legend_location`, event labels and grouped line labels; rendered group/event legend |
| Separate legend page | `event_chart_legend`; independent figure generated from the chart's actual artists, rendered example |
| Custom ticks, limits, axis suppression, titles and device layout | Returned Matplotlib Axes/Figure provide these controls; standard Python graphics replace S `par`, `mfg` and device-management options |

## Interface changes and numerical corrections

Python uses zero-based numeric indices and explicit arrays. Dataframe names and
logical row expressions are resolved by the caller before plotting; use numeric
columns or encoded sort keys for categorical sorting. Date columns can be
converted with `event_dates`. None of this requires pandas.

The converter accepts homogeneous numeric or string codes within each code
column. String sorting is deterministic Unicode order, or an explicit category
sequence; it does not depend on the current S/R locale. Missing codes are distinct
from the literal string "NA". For *line grouping*, the source explicitly excludes
empty strings and "NA", so those values suppress spans along with missing groups.
Event markers and extra interval layers are retained independently.

Source y-position indexing and filtered extra-interval defects are corrected.
The mathematically consistent Goldman boundary is the default; original slope
behavior is available explicitly. Invalid numeric range, zero native-boundary
denominators and unsupported mixed category types fail clearly. Jitter uses a
local NumPy generator; exact S random sequences are not reproduced.

Rendering uses Matplotlib symbol names and colors, ISO dates, explicit styling
vectors and configurable legends. It does not recycle underspecified style arrays,
interpret S numeric `pch` codes, pause for a graphics-device keypress, or run an S
mouse locator. Explicit legend coordinates and a separately returned legend page
replace those device interactions. Users control showing, saving, layout and any
additional formatting through the returned artifacts. Square plots include small
margins so event markers on the boundary remain visible.

## Validation boundary

The fixtures in `tests/fixtures/eventchart-*-native.json` record the original S
routines executed under R after mechanical assignment translation. Numeric and
categorical conversion ran unchanged. Only graphics callbacks were substituted
for coordinate/style capture. See the focused `tests/test_eventchart*.py` files
and [usage/differences](eventchart.md). Grouped, square and separate-legend
examples were rendered and visually inspected. No historical GUI, operating-
system font, pixel-identical image or exhaustive S-argument compatibility is
claimed. Source hashes are in `eventchart-sources.json`.
