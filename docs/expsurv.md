# EXPSURV exploratory survival analysis

Catalog entry 28 is partial. The survival-curve and inverse-survival core is
implemented, together with interactive cut-point exploration, model alignment
and linked scatterplot/survival, event-chart and censored-box views. Named
table input/output, two-sample exponential examples and GEN-DATA covariate/arrival
simulation are available. Click/rectangle selection and continuous brushing are
available across the linked views. The full coverage audit is still pending.

The source is EXPSURV version 1 from the MD Anderson catalog, distributed as
`EXPSURV_V1.tar.gz`. It contains an XLISP-STAT source file, a TeX user manual and
a readme. The source/readme permits redistribution; original code remains local
reference material and is not bundled. Source SHA-256:
`b3807d82f5e18b9d6c564642bcb6b0f785103847578e1d9d5a19581a846a2492`.

```python
from mdanderson_stats import exploratory_survival

fit = exploratory_survival([1, 2, 3, 4], [1, 1, 0, 1])
print(fit.time, fit.survival)
print(fit.at([0, 1.5, 4]))
print(fit.survival_quantile([0.75, 0.5, 0.25]))
print(fit.survival_quantile(0.5, method="source"))
```

`exploratory_survival` computes Kaplan–Meier survival. Inputs are stably sorted
without mutation. Status is binary (one=failure, zero=censoring), and omitted
status means all failures. The default groups tied observations and includes
tied censoring in the common risk set. `legacy=True` retains KMEST's sequential
per-record product and order within ties. Unlike KMEST, Python explicitly sorts
input rather than requiring callers to have applied COSORT first.

Results include event times, survival, risk counts, events and censor counts,
along with the exact left/right corners used by KM-PLOT (`step_time` and
`step_survival`). `.at()` is right continuous, with survival one before the first
observation and flat extension beyond follow-up. Zero-time failures retain their
initial vertical jump. Result arrays and query outputs are read-only.

`.survival_quantile(q)` takes a **survival level**, not a cumulative probability.
Its default step method returns the first time survival is at most q, with q=1
at time zero. An unreached level returns NaN rather than a fabricated estimate.
`method="source"` implements QUANT's linear interpolation from the bracketing
survival levels, including its unusual last-time choice at an exact plateau.
For survival [.75, .5, .5, 0] at times [1, 2, 3, 4], the default q=.5 result is 2;
the source result is 3. This interpolation is not the ordinary inverse of a step
function and is selected explicitly, independently of the tie convention.

Validation uses exact rational per-record products on censored, uncensored and
tied samples; independently computed grouped risk sets; right-continuous queries;
plot corners; interpolation and plateau examples; scaling; time-zero events;
immutability; and invalid input checks. No XLISP-STAT interpreter was available
in this environment, so these are source-formula and independent mathematical
checks, not claims of executing the archived program. Curve construction uses
NumPy sorting/aggregation/products and queries use binary searches; no measured
speedup is claimed for this increment.


## Cut-point exploration

```python
from matplotlib import pyplot as plt
from mdanderson_stats import survival_cutpoint, plot_cutpoint

data = survival_cutpoint([3, 1, 4, 2], [1, 0, 1, 1], [2, 1, 2, 3])
comparison = data.compare(2)
print(comparison.lower_indices, comparison.upper_indices)
view = plot_cutpoint(data, cut=2)
plt.show()
```

`survival_cutpoint` retains copied, aligned observations. `.compare(cut)` fits
separate survival curves for `covariate <= cut` and `covariate > cut`, preserving
original row indices. The comparison is immutable and has no plotting dependency.
An empty group has a None curve and an empty index array, rather than invented
survival estimates. Any finite cut is allowed for numerical comparisons.

With the optional plot extra, `plot_cutpoint` returns a controller containing a
figure, linked density/survival axes and a Matplotlib slider. Keep the controller
alive while interacting. The slider has 50 positions from the minimum to maximum
covariate, initially at index 25 as in the archived function. An explicit initial
cut or `view.set_cut(value)` may use any value inside the observed range. Updates
change the density marker, both curves, and group-count labels together. Empty
groups remain visible as n=0 labels with no curve. `view.close()` disconnects the
callback and closes the figure. No global backend is changed; use an interactive
Matplotlib backend for dragging, or call `set_cut` programmatically and export
with `view.figure.savefig(...)`.

The two linked panels replace the source's separate windows. The covariate
density guide uses a Gaussian kernel with `std(ddof=1)*n**(-1/5)` bandwidth,
or an explicit positive `bandwidth`. It is evaluated once in bounded chunks and
reused when moving the cut. This is an explicit replacement for XLISP-STAT's
implicit `kernel-dens` runtime dependency, not a claim of identical density
defaults. A constant covariate cannot support a varying slider and raises an
explicit error; the numerical `compare` method still supports it.

Tests verify equality at the cut, unsorted row alignment, exact group membership,
empty endpoints, immutable snapshots, source tie mode, callback updates to all
artists, initial slider synchronization, invalid-update state preservation,
density normalization/symmetry and figure cleanup. Both an interior-cut figure
and an empty-upper-group endpoint figure were rendered and visually inspected.
This is exploratory group comparison, with no automatic cut optimization or
inferential test added by the plotting workflow.


## Model-alignment views

```python
from mdanderson_stats import exploratory_survival, plot_survival_alignment

first = exploratory_survival([1, 2, 3, 4], [1, 0, 1, 1])
second = exploratory_survival([2, 4, 6, 8], [1, 0, 1, 1])
view = plot_survival_alignment(first, second, method="accelerated-failure")
view.set_parameters(1.5, 1)
```

`plot_survival_alignment` covers ACCEL-FAIL-PLOT and PROP-HAZ-PLOT with two
sliders on a shared figure. Keep the returned controller alive; use an interactive
Matplotlib backend for dragging, `set_parameters` for programmatic updates,
and `close()` to disconnect callbacks and close the figure. Plotting does not
refit the samples or mutate their curves.

For `method="accelerated-failure"`, each slider multiplies the plotted time
coordinates. Its upper limit is the larger sample maximum divided by that
sample's maximum, matching the executable source. Both start at their upper
limits so the maximum times coincide. In function notation the transformed
curve is S(t/m); the manual's S(k*t) uses the reciprocal parameter. A zero
multiplier collapses the plotted curve onto time zero. Samples whose maximum
time is zero have no defined time-scale ratio and are rejected for this view.

For `method="proportional-hazards"` (the default), sliders raise the plotted
survival values to powers from zero to one, initially one. This multiplies
cumulative hazard by the power. At power zero the display is identically one,
including the explicit plotting convention 0**0=1. Each slider offers 50 snapped
positions as in the source; programmatic values may lie anywhere in its range.
Both parameters are validated before either slider is changed.

Tests verify alignment of rescaled samples, cumulative-hazard scaling, exact
plotted coordinates, callbacks, zero endpoints, invalid-update preservation and
cleanup. A time-alignment figure was rendered and visually inspected. These are
exploratory alignment controls, not fitted regression coefficients or formal
model tests.


## Linked scatterplot selection

```python
from mdanderson_stats import plot_survival_scatter

view = plot_survival_scatter(
    [3, 1, 4, 2],
    [1, 0, 1, 1],
    [[2, 8], [1, 4], [2, 6], [3, 5]],
    labels=["Covariate A", "Covariate B"],
)
view.select([0, 2])
view.select([1], add=True)
```

`plot_survival_scatter` implements SCAT-KM with a covariate matrix figure and
separate selected-sample survival figure. Covariates have one row per patient
and one column per variable; a vector becomes a one-variable matrix. Input rows
remain aligned and are copied into read-only arrays. Initially all rows are
selected. Drag a rectangle in any matrix cell to replace the selection, hold
Shift while dragging to add rows, and press Escape to clear it. Updates happen
when a rectangle selection completes; continuous brush mode is described below.
Selected patients are highlighted in every cell, including diagonal cells.

`select(indices, add=False)` accepts original, zero-based row indices and removes
duplicates. `selected_indices` and `curve` expose the current selection and its
survival fit; `legacy=True` retains the source's sequential tie convention.
Empty selections have `curve=None` and an empty plotted line. This deliberately
clears the source's stale curve after a brush finds no patients. `clear()` also
removes rectangle outlines. Keep the controller alive for interaction and call
`close()` to disconnect callbacks and close both figures. The optional plotting
extra and an interactive Matplotlib backend are needed for mouse interaction;
programmatic selection and export work with a noninteractive backend.

Tests exercise real canvas press/move/release events, Shift-add and Escape,
original row identity, independent selected-sample survival fits, highlights in
every cell, empty selection, validation and cleanup. Matrix and survival figures
were rendered and visually inspected. As with the other EXPSURV views, validation
uses source inspection and independent checks, not an archived XLISP-STAT run.


## Linked event charts

```python
from mdanderson_stats import plot_event_scatter

view = plot_event_scatter(
    arrival=[7, 1, 4],
    duration=[3, 8, 0],
    status=[1, 0, 1],
    covariates=[[0, 0], [1, 1], [2, 2]],
)
view.select([0, 1])
```

`plot_event_scatter` implements SCAT-EVENT with the same matrix controls and
original-row selection API as `plot_survival_scatter`. The event chart plots
horizontal segments from (0, arrival) to (duration, arrival). Duration is observed
follow-up, not the calendar endpoint arrival+duration. A plus marks failure
(status=1); an open diamond marks censoring (status=0). Zero-duration observations
retain an endpoint marker. Arrival and duration must be finite, nonnegative and
aligned with binary status and covariate rows. Inputs are copied and read-only.

The controller exposes `event_figure`, `event_axes`, a `segments` LineCollection,
and separate `failures` and `censored` endpoint collections. Updates replace
collection coordinates in batches rather than creating a line artist per patient.
Axes retain the full-sample ranges while selecting; zero maxima use a display
range of 0 to 1. Matplotlib supplies tick placement instead of XLISP-STAT's
get-nice-range routine. The event chart initially shows all patients, making it
consistent with the initial matrix highlights; the source leaves this chart blank
until the first selection. Empty selections clear all segments and markers,
including after a rectangle finds no observations. This fixes the source's stale
empty-brush display. Continuous brush mode is described below.

`clear()` clears both selection and rectangles; `close()` disconnects matrix
callbacks and closes both windows. Tests check exact source coordinates, failure
and censoring partitions, unsorted original rows, additive selection, fixed axes,
zero times, immutable inputs, invalid input and update preservation, and actual
canvas selection/Escape events. An event chart was rendered and visually inspected.
These are source-formula checks, not native XLISP-STAT execution or a measured
rendering-speed comparison.


## Censored survival boxes

```python
from mdanderson_stats import censored_box, plot_censored_box

box = censored_box([1, 2, 3, 4], [1, 1, 0, 0])
print(box.quartiles, box.last_failure_survival)
view = plot_censored_box([1, 2, 3, 4], [1, 1, 0, 0], [0, 1, 2, 3])
view.select([0, 2])
```

`censored_box` implements the numerical SCAT-BOX geometry without requiring
Matplotlib. Quartiles are returned in ascending cumulative-probability order
(.25, .5, .75), obtained from survival levels (.75, .5, .25). The default
`quantile_method="source"` preserves QUANT interpolation and last-plateau
semantics; `quantile_method="step"` selects ordinary step-curve inverse queries.
`legacy=True` independently selects sequential, rather than grouped, ties.

The immutable result retains its fitted curve, read-only quartiles and line
segments, and the first/last failure times with survival immediately after those
failures. Unreached quartiles are NaN. Horizontal bars mark available quartiles;
side walls span the lower to upper quartile when all exist, otherwise extend to
the last failure. If no quartile is available, walls span first to last failure.
The source's horizontal coordinates (.667, 1.33) are preserved. For all-censored
samples the failure metadata is None and segments are empty, handling a case
where the archived code indexes an empty death-time list.

An exact survival plateau can put a source-interpolated quartile beyond the last
failure. For failures at 1, 2, 3, 4 followed by censoring at 5, 6, 7, 8, the source
median is 8 while the unfinished walls stop at 4. This unusual geometry is
preserved deliberately; step mode gives median 4. The plot does not substitute
raw-data quantiles, extrapolate unreached quartiles, or add ordinary-boxplot
outlier fences.

`plot_censored_box` links the common rectangle/Shift-add/Escape matrix controls
to this geometry. It exposes `summary`, `box_figure`, `box_axes`, `segments` and
an annotation of survival at the last failure. All observations start selected;
`select` accepts original row indices. A selection with no failures retains a
summary and displays “No observed failures”; an empty selection has summary=None
and displays “No selection”. `close()` disconnects callbacks and closes both
figures. Fixed full-sample vertical limits include zero and a small top margin;
Matplotlib supplies ticks instead of the source runtime. Continuous brushing
uses the common controls described below.

Tests independently cover all source drawing branches, plateau quartiles,
explicit step mode, first/last failure survival with ties, zero-time events,
immutable outputs, actual canvas selection, invalid-update preservation,
all-censored/empty transitions and cleanup. Partial and all-censored figures
were rendered and visually inspected. Validation remains source inspection and
mathematical checks, not execution of the archived XLISP-STAT interpreter.


## Named tables and exponential examples

```python
from mdanderson_stats import ExploratoryTable, generate_exponential_samples

table = ExploratoryTable.read("patients.txt", ["time", "status", "age"])
ordered = table.cosort("time")
ordered.write("patients-sorted.txt")
first, second = generate_exponential_samples(rng=42)
```

`ExploratoryTable` replaces GET-DATA, ASSIGN-VARS/MYSET, COSORT and SAVE-DATA's
PRINT-LINE/PRINT-LIST path with explicit names and a copied read-only numeric
matrix. `column(name)` retrieves an aligned read-only column; unknown names raise
KeyError. `cosort(name)` returns a new table, with stable ties. No Python globals
are assigned and no file dialogs are opened. Names must be unique and nonempty.

Files have no header, one observation per nonblank line, and whitespace-separated
numbers. Missing/nonfinite values, ragged rows and nonnumeric text are rejected;
parse errors identify the line. Blank lines are ignored, and empty files are
rejected. Names are supplied separately, matching the original GET-DATA arguments.
`write` uses 17 significant digits for float64 round trips, UTF-8 and newline
separators; it replaces the requested file. This preserves the numeric format
rather than Lisp's exact whitespace or number-printing defaults. It does not
interpret Lisp expressions or silently treat comments as data.

`generate_exponential_samples` covers GEN-EXPO-DATA; its defaults (100 and 50
observations, censoring probability .1, rates 1 and 10) cover GEN-EXPO-EXAMPLE.
It returns two sorted tables with `time` and binary `status` columns. Exponential
parameters are **rates**; NumPy receives their reciprocal scales. Status is drawn
independently of latent lifetime. For status zero, observed follow-up is a
uniform fraction of that latent lifetime. Consequently this illustrative source
mechanism is not ordinary independent censoring-time simulation. The expected
observed time is (1-p/2)/rate and its second moment is (2-4p/3)/rate².

An integer `rng` seed replays results in the same NumPy environment; a supplied
Generator advances its state, and None creates a fresh generator. NumPy replaces
the XLISP-STAT random stream, so source seed equivalence and cross-version bitwise
stability are not promised. See the [NumPy generator documentation](https://numpy.org/doc/stable/reference/random/generator.html)
and [exponential parameterization](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html).
Inputs are validated before drawing. Sample sizes and rates must be positive;
nonrepresentable scales and generated overflow fail explicitly. Random draws,
censoring transformations and sorting are batched with NumPy; no speedup against
the archived interpreter is claimed.

Tests exercise real file round trips at float64 precision, whitespace, single-row
and single-column files, malformed input, stable ties and input immutability.
Generator checks cover replay, state advancement, rate scaling, no/all censoring,
invalid-input state preservation, and fixed-seed large samples against independent
analytic moments. These are mathematical and integration checks, not archived
XLISP-STAT runs.


## Covariate and arrival simulation

```python
from mdanderson_stats import generate_exploratory_data, plot_event_scatter

data = generate_exploratory_data(200, rng=12)
view = plot_event_scatter(
    data.column("arrive"),
    data.column("length"),
    data.column("status"),
    data.values[:, 3:],
    labels=["x", "y", "z"],
)
```

`generate_exploratory_data(n, rng=...)` implements GEN-DATA. It draws independent
uniform x and y, sets z=x*y, and draws latent exponential survival with rate
10*x+y. Lifetimes are divided by their sample standard deviation. Independent
arrival times are uniform on [0,2); observed follow-up ends at death or the fixed
study close at 2. Deaths exactly at 2 are censored, matching the executable source's
strict comparison. The returned `ExploratoryTable` contains `length`, `arrive`,
`status`, `x`, `y`, `z`, stably sorted by length with all columns aligned.

The sample standard deviation uses denominator n−1, verified in
[XLISP-STAT stats.lsp](https://github.com/jhbadger/xlispstat/blob/f1bea6053df658ee48612bf1f63c35de99e2c649/src/lsp/stats.lsp#L146).
NumPy uses the corresponding `ddof=1`. To avoid overflow from squaring large
latent lifetimes, the implementation first divides them by their maximum and
then divides by that scaled sample's standard deviation. This yields the same
normalized lifetimes mathematically. Degenerate or nonfinite lifetimes raise an
explicit error. At least two observations are required. Rescaling by a statistic
of the whole sample makes the normalized lifetimes dependent; these are the
source's exploratory examples, not an independent exponential sample after
normalization.

The RNG contract is the same as the two-sample generator: explicit seed or
Generator, with NumPy streams replacing XLISP-STAT streams. Uniform generators
may produce zero; a generated zero rate fails rather than fabricating a lifetime.
No original interpreter or random sequence is bundled or claimed to have run.

Tests compare normalization and censoring to independent scalar calculations,
check invariance under an enormous latent scale change, exercise the exact
study-close boundary, reject undefined sample variation, verify seeded replay,
uniform covariate moments, z=x*y, censoring endpoints, immutable rows and aligned
sorting, and preserve RNG state on invalid sample sizes.


## Common selection and continuous brushing controls

All three matrix controllers support the source's selection and brushing modes.
The initial mode is `select`: click a patient or drag a rectangle to replace the
selection; hold Shift to add. Clicking blank space clears it. Clicks select all
points within six display pixels, including coincident points; motion of at most
three pixels is treated as a click. This explicit tolerance replaces runtime
point-hit defaults. Press Escape to clear the selection.

Press B to toggle continuous brushing, or call
`view.set_selection_mode("brush")` / `view.set_selection_mode("select")`.
In brush mode a dashed rectangle follows the mouse inside any matrix cell and
replaces selection on every motion event. No mouse button is required. Moving
into an empty region clears the linked display. Leaving the matrix hides the
brush while retaining the latest selection. Returning to selection mode restores
rectangle controls. The brush is clipped visually to its current axes.

The default brush is 40 by 40 display pixels. `view.set_brush_size(width, height)`
sets finite positive dimensions independently of covariate scales. Plus (or =)
and minus enlarge/shrink both dimensions, with keyboard sizes bounded to 1–4096
pixels. Resizing hides the old outline; the next motion uses the new size.
These controls replace the source's mouse-mode and brush-resize menu dialogs.
No browser or global Matplotlib backend is required or changed.

Selection uses transformed covariate coordinates in a NumPy batch. If the row
set is unchanged, the linked fit and point colors are reused, avoiding repeated
survival calculations as the brush moves within the same group. `close()` removes
all added mouse/key callbacks and closes both figures. Tests drive actual mouse
and keyboard events across survival, event and censored-box controllers, covering
hover replacement, empty regions, mode switching, resize, click, Shift-add,
blank clicks, invalid controls, repeated-row-set reuse and cleanup. A rendered
brush selection was visually inspected. These controls require an interactive
Matplotlib backend for live use; automated canvas tests use Agg.
