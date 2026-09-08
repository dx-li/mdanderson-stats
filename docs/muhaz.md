# MUHAZ kernel hazard estimation

Catalog entry 49 is **implemented**. The fixed-bandwidth kernel calculation is
implemented, together with piecewise-exponential estimates and their numerical
reports, and Nelson/product-limit failure-interval estimates. Global, local and
nearest-neighbor bandwidth selection, bandwidth smoothing and candidate
bias/variance/MSE diagnostics and structured summaries are implemented. Kernel,
piecewise and stratified plots are also implemented. The completed
[source coverage audit](muhaz-coverage.md) maps the archive to implementation and
validation evidence.

Source: [MUHAZ version 1](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/49),
distributed as `MUHAZ_V1.tar.gz`. Its archive contains `muhaz.f`, the S interface
`all.s`, help files and installation machinery. The kernel method references
Mueller and Wang (1994), Biometrics 50:61–76. Original source and binaries are
local reference inputs and are not bundled.

```python
from mdanderson_stats import muhaz_fixed

fit = muhaz_fixed(
    [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3.0],
    [1, 0, 1, 1, 0, 1, 0],
    bandwidth=0.75,
    kernel="epanechnikov",
    boundary="both",
)
print(fit.time, fit.hazard)
```

`times` is a nonempty vector of finite nonnegative follow-up times. `delta`
contains 0 for censoring and 1 for events; omission means all events. Bandwidth
must be a positive finite scalar. Bounds default to `(0, max(times))` and must
have positive length. The default evaluation grid has 101 points; an explicit
grid preserves order and must lie within the bounds. An empty grid is valid.
Arrays in `MuhazFixed` are read-only snapshots.

The four kernels are `rectangle`, `epanechnikov`, `biquadratic` and `triquadratic`.
Correction is `none`, `left` or `both`. Boundary kernels have unit integral and
zero first moment on their truncated support. Right correction reflects the
left kernel. When a bandwidth spans both boundaries, the archive gives left
correction priority; that convention is retained. This is a one-sided boundary
construction, not a separately derived two-boundary kernel for that situation.
Negative totals from the signed boundary kernels are clipped to zero, as in the
source. This clipping means the final estimator need not retain every linear
kernel identity.

## Ties and exact support endpoints

The default weights each distinct event time by d/r, where d counts tied events
and r includes every observation still at risk at that time, including tied
censoring. This gives permutation-invariant Nelson increments. Kernel support is
closed, including exact endpoints; boundary correction also enforces its physical
truncation at the selected bound.

`legacy=True` reproduces archived HAZDEN instead: event weights are `1/(n-i+1)`
after stable time sorting. Censoring interspersed among ties therefore changes
weights, and multiple tied events get sequential denominators. IBNDS also excludes
the left support endpoint and generally the right endpoint, except when the upper
support reaches the final observation. This particularly affects rectangle
kernels. These source behaviors are explicit compatibility options, not silently
used by the default estimator. Legacy left correction can also retain observations
before the specified left bound, whereas the default enforces the truncated
kernel support. All observations remain in the risk-set calculation.

This core uses maximum follow-up by default, rather than the high-level S
interface's interpolated ten-at-risk bound. It requires an explicit bandwidth
and does not imply that a bandwidth has been optimized. The forthcoming complete
MUHAZ workflow will address bandwidth selection and its settings separately.

## Validation and computation

`tools/reference_muhaz_fixed.py` compiles unchanged HAZDEN, IBNDS and KERNEL with
an independent driver. Thirty-six native curves span four kernels, three boundary
settings, untied/censored and tied observations, exact support endpoints, and
bandwidths exceeding the domain width. Fixtures record numerical source,
extracted-source and driver hashes, and compiler version. They validate legacy
mode, which is the mode that reproduces those archived routines.

Independent tests validate unit mass and zero first moment using polynomial
quadrature, grouped-tie d/r increments, permutation invariance, closed support,
inverse time-scaling of hazards, clipping of negative boundary totals, empty
queries, all-censored data and invalid inputs. A large grid is checked against
separate evaluations to exercise chunked computation.

NumPy evaluates kernels over both grid and event axes. Only event times enter
smoothing; censored observations still determine risk sets. Grid chunks bound
intermediate memory instead of allocating the entire grid-by-subject matrix.
For M grid points and D distinct event times (individual events in legacy mode),
smoothing takes O(MD) work after sorting; working storage is O(N+D+M) plus a
bounded number of grid rows by D. No measured speedup over Fortran is claimed.

## Piecewise-exponential estimates

```python
from mdanderson_stats import pehaz

bins = pehaz([0.5, 1, 2, 2.5, 3], width=1, bounds=(0, 2.5))
print(bins.report())
# bins.write_report("piecewise-hazard.tsv", digits=17)
```

`pehaz` calculates constant hazard in each bin as the number of events divided
by total person-time. All subjects are observed from time zero until their
recorded event/censoring time; delayed entry is not an input. `delta` is optional
and defaults to all events. Bounds default to zero and maximum follow-up. The
result retains requested bounds, width, actual bin cuts, hazard, event counts,
left-endpoint risk counts and person-time in read-only arrays. Bin summaries
and UTF-8 file reports expose these sufficient statistics without refitting.

With `width=None`, the source formula is `(right-left)/(8*events**0.2)`, using all
input events. An all-censored dataset therefore requires explicit width rather
than inventing a default. Every width must be positive and finite; bounds must
have positive length. Widths too small to produce distinct floating-point cuts
are rejected.

Default bins are left-closed/right-open, except that the final bin includes its
right endpoint and stops exactly at the requested maximum. An event at an
interior cut belongs to the bin starting there. Risk counts include subjects
ending at the left cut, even though those subjects contribute zero person-time
to that bin. Subjects continuing beyond the final cut contribute exposure up to
that cut. Data before the requested lower bound contribute neither events nor
exposure within the analysis window.

`legacy=True` reproduces the archived `pehaz` bin conventions: equal-width bins
can extend beyond the requested maximum, and every right endpoint is excluded.
Consequently, events beyond the requested maximum can enter an overshooting
last bin, while an event exactly on the final cut is omitted. These behaviors
are documented compatibility choices rather than the default.

A bin with no exposure and no events has NaN hazard, printed as NA because its
hazard is unidentifiable. Events with zero exposure yield positive infinity,
representing an unbounded likelihood. A finite hazard exceeding floating-point
range raises a numerical error; it is not confused with the zero-exposure case.
The data object retains these distinctions; `plot_pehaz` displays nonfinite
bins as gaps.

After sorting, binary searches locate each bin's observations and cumulative
event counts supply totals. Person-time is accumulated from disjoint slices of
ending observations plus the full-bin exposure of continuing subjects. This
uses O(N+B) storage and O(N log N + B log N + N + B) work for N observations and
B bins, rather than an N-by-B matrix. Local time differences avoid cancellation
from subtracting large cumulative follow-up sums.

`tools/reference_pehaz.py` executes the unchanged archived S function in R;
10 fixtures cover censoring, ties, endpoint events, nonzero lower bounds,
overshooting bins, no-exposure bins and automatic width. Source/extraction/driver
hashes and R version record provenance. Independent tests check person-time
conservation, individual bin counts, final-boundary semantics, permutation and
time scaling, precision at large time origins, extreme width/domain ratios,
report round trips, undefined/infinite hazards and invalid inputs.

## Failure-interval hazard estimates

```python
from mdanderson_stats import kphaz

fit = kphaz([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], q=2, method="nelson")
print(fit.time, fit.hazard, fit.variance)
```

`kphaz` implements the archive's `kphaz.fit` workflow. Within each stratum,
let t[j] denote the sorted distinct failure times. The estimate at the midpoint
of t[j] and t[j+q] is the cumulative-hazard increment over that interval divided
by its duration. Its variance is the corresponding cumulative-variance increment
divided by squared duration. The event at the interval's left endpoint is excluded;
there is no initial interval from zero to the first failure. A stratum with at
most q distinct failure times has no output rows. All-censored input is rejected,
while an all-censored stratum within a dataset with events remains listed with
no rows.

The default Nelson event increment is d/r with variance increment d/r². The
product-limit increment is −log(1−d/r), with Greenwood increment d/[r(r−d)].
Here d counts tied failures and r includes all subjects at risk immediately
before the time, including tied censoring. These grouped increments are invariant
to input order within ties. If the entire remaining risk set fails, the
product-limit hazard and variance are infinite, and that result is retained.

`legacy=True` uses the source's sequential risk denominators after stable time
sorting. It also retains the product-limit variance's 0/0 term when the final
observation is a censor tied with a failure, producing NaN in the affected
interval. The default grouped estimator avoids this artifact. Status must be
exactly 0/1 and q a positive integer; the source's coercion of other status values
and truncation of fractional q are not performed. Missing stratum labels are
rejected instead of relying on the archive's inconsistent NA indexing behavior.
Method names must be complete, without implicit partial matching.

The immutable result contains midpoint times, hazard, variance, a zero-based
`stratum_index` into its `strata` label tuple, q, method and compatibility mode.
Strata can be numeric or string labels, following the package's existing label
validation. Within each stratum, output is ordered by interval start time.

`tools/reference_kphaz.py` executes the source numerical calculation in R after
two documented S syntax adaptations: `is.inf` becomes `is.infinite`, and
multi-value `return` expressions become named lists. Fixtures record source,
extracted/adapted code and driver hashes plus the R runtime. Twenty-four cases
cover both methods, three q values, stratification, ties, terminal failures,
terminal tied censoring, and empty output. Independent tests verify reciprocal
Nelson hazards, log-survival ratios, Greenwood increments, grouped ties,
permutation/time scaling and immutable outputs.

Sorting and aggregation take O(N log N) work. Consecutive event increments are
summed with prefix arrays, using extended precision for finite terms and separate
counts of infinite/NaN terms. This avoids subtracting infinite cumulative totals
and repeated scans of all earlier subjects. Window evaluation takes O(M) work
and storage for M distinct failure times, independently of q. `plot_kphaz`
provides stratified step plots with nonfinite estimates shown as gaps.

## Candidate-bandwidth diagnostics

```python
from mdanderson_stats import muhaz_mse

comparison = muhaz_mse(
    [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3.0],
    [1, 0, 1, 1, 0, 1, 0],
    bandwidths=[0.35, 0.8, 2.0],
    pilot_bandwidth=0.65,
)
print(comparison.mse)  # bandwidth rows, evaluation-time columns
print(comparison.converged)
```

`muhaz_mse` implements the archived MSEMSE convolution calculation. Its pilot
hazard and candidate kernel give a convolution bias and variance estimate;
MSE is squared bias plus variance. The source survival factor is exactly
`1 - number_of_events_at_or_before_time/(N+1)`. This differs from Kaplan–Meier
survival and from survival conditional on being uncensored. It is retained as
part of this specific bandwidth-selection criterion.

The default quadrature uses successively refined trapezoids, stopping when both
integrals meet relative tolerance 0.001 or after six refinement levels (at most
33 quadrature points). These match the archived settings. `max_refinements`
can be 1–16, and `rtol` can be any finite nonnegative value. The read-only result
exposes bias, variance, MSE, the pilot hazard, refinement counts and per-cell
convergence flags, plus the numerical settings. Reaching the refinement cap is
reported as unconverged, rather than being silently treated as accurate. This
relative-change check is a numerical heuristic, not a rigorous error bound.

`bandwidths` accepts a positive scalar or nonempty vector, preserving order and
duplicates. Grid/bounds/pilot validation follows `muhaz_fixed`; the default grid
is 101 points and default upper bound is maximum observed follow-up. This is
explicit diagnostic input, not yet the archived high-level selector's default
minimization grid or ten-at-risk bound. No bandwidth is selected here.

Default tie/support conventions follow `muhaz_fixed`. `legacy=True` reproduces
archived sequential weights, support endpoints, and FUNC's unconditional kernel
reflection near the right boundary even when an overlapping left correction
was chosen. The default reflects only when right correction is selected.
Candidate and pilot bandwidths remain separate. Pilot convolutions can require
queries outside the output grid; the prepared hazard evaluator handles these
without repeatedly sorting the input or reconstructing risk sets.

`tools/reference_muhaz_mse.py` invokes unchanged Fortran MSEMSE and dependencies.
Twenty-four grids cover all four kernels, all three boundary settings, tied and
untied samples, three candidate bandwidths and nine time points: 648 comparisons
of bias, variance and MSE. Source/driver hashes, compiler and flags are recorded.
The oracle uses `-ffp-contract=off`: fused multiply-subtract operations can place
an exact-boundary quadrature point on a different side of a discontinuity and
materially change the archived correction. NumPy's separate operations are
compared to separate Fortran operations. This documents reproducibility limits;
it does not promise agreement with every compiler's fused-arithmetic output.

Independent checks include a constant-pilot rectangular convolution with known
integrals, empty queries, zero-event convergence, explicit refinement exhaustion,
MSE decomposition, candidate ordering and inverse time scaling. All earlier fixed
hazard tests also pass after extracting the shared prepared evaluator. Quadrature
samples and pilot hazard kernels are evaluated in chunks to avoid a full
bandwidth-by-time-by-quadrature-by-subject allocation. Global, local and nearest-neighbor selection use these diagnostics.

## Global bandwidth selection

```python
from mdanderson_stats import muhaz_global

fit = muhaz_global(
    [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3.0],
    [1, 0, 1, 1, 0, 1, 0],
    bandwidths=[0.35, 0.8, 2.0],
    pilot_bandwidth=0.65,
    bounds=(0, 3),
)
print(fit.bandwidth, fit.scores)
print(fit.diagnostics.converged)
```

`muhaz_global` connects the MSE criterion to one bandwidth shared across the
estimated curve. Candidate scores are **sums of MSE at minimization-grid points**,
as in GLMIN. They are not quadrature-weighted integrals and depend on the number
of grid points. The default chooses the first candidate attaining the minimum,
including zero. Both the selected index and full scores are retained, together
with the complete MSE/convergence diagnostics. The final curve is evaluated at
the selected bandwidth; selection does not imply that quadrature converged.

`legacy=True` reproduces the original positive-score rule: only scores strictly
between zero and 1e30 qualify. If none qualify, it selects the last candidate and
reports the source sentinel score 1e30, even if that candidate's computed score
is zero. Actual score arrays remain available. Contrary to its comment, GLMIN's
executable code does not discard a minimum at the first bandwidth; Python follows
the executable behavior. These are grid minima, not continuous global optima.

A single candidate bypasses MSE, matching NEW_HAD. `scores`, `score` and
`diagnostics` are then None, rather than exposing the source's uninitialized
outputs. The bandwidth is used directly, and no pilot bandwidth is required.

With bounds omitted, the upper time is linearly interpolated at a risk count of
ten across distinct observed follow-up times. Counts include both events and
censoring. If ten is outside the available risk-count range (for example, fewer
than ten observations), explicit bounds are required. The lower time is zero.
An explicit upper bound beyond maximum follow-up is truncated to that maximum,
as in the S interface, and effective bounds remain accessible in `fit.curve`.
An optional boolean `subset` is applied jointly before selected-data validation.

Without a supplied pilot, the default is `(right-left)/(8*events**0.2)`; legacy
mode uses the S code's `right/(8*events**0.2)`, which differs for a nonzero lower
bound. Without supplied candidates, 25 equally spaced values from 0.2 to 20 times
the pilot are generated. Twenty-five follows the executable S code, whose help
text instead says 21. The default grids have 51 minimization and 101 estimation
points; positive integer `n_min_grid` and `n_est_grid` control their sizes. An
all-censored sample needs an explicit pilot for multiple-candidate selection.
No smoothing is applied to the selected common bandwidth.

`tools/reference_muhaz_global.py` calls unchanged NEW_HAD and GLMIN with the
recorded non-fused compiler flags. Thirty-nine native fits cover all kernel and
boundary combinations, censoring, ties, zero-score fallback and single-candidate
bypasses, comparing selected bandwidths, all candidate scores, selected scores,
and fitted hazards. Independent tests verify score minimization, selected-curve
agreement, default formulas, risk-count interpolation, subsets, bound truncation,
ties in scores, immutable candidates and input validation. A 1,001-point grid is
also tested, exceeding the archived fixed pilot buffer.


## Local bandwidth selection and smoothing

```python
from mdanderson_stats import muhaz_local

fit = muhaz_local(
    [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3.0],
    [1, 0, 1, 1, 0, 1, 0],
    bandwidths=[0.35, 0.8, 2.0],
    pilot_bandwidth=0.65,
    smoothing_bandwidth=0.8,
    bounds=(0, 3),
)
print(fit.local_bandwidth)  # Selected at fit.diagnostics.time
print(fit.bandwidth)  # Smoothed bandwidth used at fit.time
print(fit.hazard)
```

`muhaz_local` selects the first candidate attaining the smallest MSE separately
at each minimization-grid point. `minimum_mse`, `selected_bias`, and
`selected_variance` describe those selected candidates; `selected_index` indexes
the candidate rows in `diagnostics`. `score` is the sum of these pointwise minima,
not the MSE of the final smoothed-bandwidth curve. The full candidate diagnostics
retain quadrature refinement counts and convergence flags.

The selected bandwidths undergo kernel regression onto the estimation grid,
using the same kernel family and a smoothing bandwidth defaulting to five times
the pilot. The resulting bandwidth at each estimation point is used in its
hazard calculation. NumPy evaluates varying bandwidths across both time and event
axes in bounded chunks; it does not repeatedly sort the sample or call the public
fixed estimator for each point. Smoothing also bounds intermediate storage.

Subsets, effective bounds, ten-at-risk interpolation, pilot defaults, candidate
defaults, and grid sizes are shared with `muhaz_global`. A single supplied
candidate bypasses both selection and smoothing: `bandwidth` is constant and
local-selection fields, `score`, `diagnostics`, and `smoothing_bandwidth` are
None. No pilot is required in that case.

By default, zero MSE is eligible and “left” correction affects only the left
boundary. `legacy=True` retains two executable source conventions:

- LOCLMN accepts only `0 < MSE < 1e30`. If none qualify at a point, it chooses the
  last candidate and records the sentinel minimum 1e30. Its bias and variance
  outputs are uninitialized; Python reports NaN for those unavailable entries.
  The source's apparent zero-score assignment is overwritten by the sentinel.
- BSMOTH applies both boundary corrections when requested to correct only the
  left boundary. This differs from HAZDEN's handling of the same setting.

Both modes retain left-boundary precedence when correction regions overlap.
Boundary kernels may have negative weights, so smoothed bandwidths are not
necessarily confined to the candidate range. An uncovered smoothing point,
zero denominator, nonfinite arithmetic, or nonpositive resulting bandwidth
raises a clear RuntimeError. Python does not replace it with an invented
bandwidth or silently clip it. Revise the smoothing bandwidth or grid settings.

`tools/reference_muhaz_local.py` runs unchanged NEW_HAD, LOCLMN and BSMOTH with the
recorded non-fused compiler flags. Its 117 native cases cover all four kernels,
three boundary settings, censored and tied samples, no-event fallback, the
single-candidate bypass, and three smoothing widths including overlapping
boundary regions. All defined local bandwidth, smoothed bandwidth, score,
bias/variance and hazard outputs are compared. The driver emits zero placeholders
for uninitialized bias/variance cells; tests exclude those placeholders and
separately require Python's explicit NaNs.

Independent tests check arithmetic-mean smoothing for an interior rectangle
kernel, constant-bandwidth preservation, left-only correction, pointwise
minimization, selected fixed-fit agreement, time scaling, shared defaults,
subsetting, immutable arrays, undefined/negative smoothing, and variable-bandwidth
evaluation across chunk boundaries.


## Nearest-neighbor bandwidths and fitting

```python
from mdanderson_stats import muhaz_knn, muhaz_neighbor_bandwidths

times = [0.2, 0.4, 0.8, 1.1, 1.7, 2.1, 2.4, 2.8, 3.0]
radii = muhaz_neighbor_bandwidths(times, neighbors=[2, 3, 4], grid=[0, 1, 2, 3], method="failures")
fit = muhaz_knn(
    times,
    neighbors=[2, 3, 4],
    pilot_bandwidth=0.65,
    smoothing_bandwidth=1.3,
    bounds=(0, 3),
)
print(fit.neighbors, fit.scores)
print(fit.bandwidth, fit.hazard)
```

`muhaz_neighbor_bandwidths` exposes both archived bandwidth constructions without
fitting a hazard. Rows correspond to requested neighbor counts, preserving order
and duplicates; columns correspond to `grid`:

- `method="failures"` returns the kth smallest absolute distance to an observed
  failure. Censored records do not enter this distance order statistic. Tied
  failures can give zero bandwidth. Counts cannot exceed the number of failures.
- `method="survival"` uses the ONEOLF survival-mass radius rule and is the default,
  matching the S interface's choice. It builds a Kaplan–Meier table at distinct
  follow-up times, including censor-only times. Trials are sorted distances in
  the archived index window around each grid point. It compares
  `S(z-r)-S(z+r)` against `1.00001*(k-1)/N`, then applies the archived radius
  adjustments by factors 1.00001 and 0.99999. Both survival lookups are right
  continuous: the interval excludes its left endpoint and includes its right.
  The source comment's left-limit notation at the right endpoint does not match
  its GETS implementation. User-supplied counts may be at most N.

The survival method retains the executable search-window and perturbation rules;
it does not claim to solve a continuous, unrestricted radius optimization.
`legacy=True` additionally reproduces KAPMEI's omission of the last observation
when its time is unique. The default includes that terminal time. A terminal tie
is included in both modes. The archived one-subject table is empty and raises an
explicit error in legacy mode. Both methods require at least one failure.
Zero radii are valid outputs of the bandwidth-only function; negative or nonfinite
outcomes are rejected.

Distance selection uses NumPy partitioning in the failure-count method. The
survival method constructs its Kaplan–Meier table once, then batches distance
sorting, survival lookups and radius decisions. Both bound intermediate storage
by processing grid chunks and have no 20,000-observation source-buffer limit.

`muhaz_knn` fits the complete curve. Its default neighbor candidates are the
integers from 2 through `floor(events/2)`; fewer than four failures require an
explicit count or candidate vector. It shares bounds, subsetting, grid and pilot
conventions with the other selectors. For each candidate count, it scores that
count's varying bandwidths by summed MSE over the minimization grid, chooses the
first minimum, and smooths the chosen radii before estimating the hazard.
`neighbor_bandwidths` retains every candidate radius; `diagnostics` retains every
MSE and convergence flag. `scores` describe the unsmoothed candidate curves.
The smoothing bandwidth defaults to five times the pilot.

A single neighbor count bypasses MSE selection, with `score`, `scores` and
`diagnostics` set to None, but still smooths the radii. With multiple candidates,
zero radii are rejected before MSE division; revise the count candidates or grid.
The existing smoothing errors apply to negative or undefined smoothed radii.
The legacy selector's initial score cutoff is 1e5; if no candidate beats it,
Python raises an error instead of using the source's uninitialized optimal count.
The default has no arbitrary cutoff.

`muhaz_mse` also accepts a candidate-by-time bandwidth matrix for these varying
candidates; a scalar or vector keeps the existing constant-bandwidth meaning.
Legacy quadrature reproduces TRY's repeated addition of quadrature abscissas,
including direct use of the interval endpoints. This matters when rounding
moves a pilot query across a discontinuity. Default mode uses direct abscissa
formulas. Existing fixed, global and local reference comparisons still pass.

`tools/reference_muhaz_neighbors.py` records 24 native KNNCEN/OLAFBW grids,
including ties, censoring, endpoint queries and extrapolation. The complete
KNNHAD oracle in `tools/reference_muhaz_knn.py` records 96 fits across both methods,
four kernels, three boundary settings, two samples, and selection/bypass cases.
Ninety-two produce valid positive smoothed bandwidths and match selected counts,
radii, scores and fitted hazards. Four produce negative smoothed bandwidths;
tests verify explicit rejection instead of treating those source hazards as valid.
Source/driver hashes and compiler flags accompany both fixture files.
Independent checks cover distance order statistics, survival endpoint conventions,
the terminal-time correction, scaling, subsetting, default candidates, MSE
matrix/scalar agreement, selected-curve agreement, immutable arrays and input
errors. A 25,001-observation case exceeds the original static buffer, and both
bandwidth algorithms are checked across chunk boundaries.


## Summaries and text reports

```python
from mdanderson_stats import muhaz_global, summarize_muhaz

fit = muhaz_global(
    [0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3.0],
    [1, 0, 1, 1, 0, 1, 0],
    bandwidths=[0.35, 0.8, 2.0],
    pilot_bandwidth=0.65,
    bounds=(0, 3),
)
summary = summarize_muhaz(fit)
print(summary.report())
summary.write_report("hazard-summary.txt")
```

`summarize_muhaz` accepts global, local and nearest-neighbor fits and returns an
immutable `MuhazSummary`. It covers the archived `summary.muhaz` fields: selected
sample and censoring counts, method, kernel, boundary correction, effective time
bounds, grid sizes, pilot/smoothing settings, chosen constant bandwidth or
neighbor count where applicable, and the minimization score. Results now retain
pilot settings and the requested minimization-grid size even when MSE selection
is bypassed. Older manually constructed result objects can omit this metadata;
unknown grid sizes are reported as “not retained”.

The report uses significant digits (`digits=6` by default, 1–17 supported), so
small positive bandwidths are not printed as zero by rounding to two decimal
places. It identifies legacy conventions and reports converged candidate/grid
cells when MSE was computed. A single-candidate bypass is labeled explicitly and
has no invented MSE score. A single-neighbor fit still reports its smoothing
settings; a constant-bandwidth bypass has no smoothing operation.

Scores retain their numerical meaning: a local score is the sum of pointwise
minima before bandwidth smoothing; global and neighbor scores are summed grid
MSE for the selected candidate. No report calls these quadrature-weighted
integrals or treats them as the MSE of a final smoothed-bandwidth curve.
`report()` returns text without printing; `write_report()` writes UTF-8 and
propagates filesystem errors. Tests cover selected-data metadata, bound
truncation, all three selectors, convergence exhaustion, bypasses, legacy
sentinels, significant-digit output, immutability, and file round trips.


## Hazard plots and overlays

```python
from mdanderson_stats import muhaz_global, pehaz, plot_muhaz, plot_pehaz

times = [0.2, 0.4, 0.8, 1.1, 1.7, 2.1, 2.4, 2.8, 3.0]
fit = muhaz_global(times, bounds=(0, 3))
ax = plot_muhaz(fit, label="Kernel")
plot_pehaz(pehaz(times, width=0.5), ax=ax, label="Piecewise")
ax.legend()
ax.figure.savefig("hazard.png")
```

The optional `plot` extra provides `plot_muhaz`, `plot_pehaz` and `plot_kphaz`.
Each returns Matplotlib axes without showing or saving the figure or changing
global style/backend. Supplying `ax` adds an overlay and retains its labels and
manual axis limits. Returned artists/axes support further customization.
`plot_muhaz` accepts fixed, global, local and nearest-neighbor kernel fits and
plots their actual evaluation coordinates. `plot_pehaz` draws constant hazards
between actual bin edges, including the corrected shortened final bin or legacy
overshoot. It draws no artificial endpoint drop to zero. These overlays cover
the archived `lines.muhaz` and `lines.pehaz` workflows.

`plot_kphaz` draws a separate step curve for each stratum, with distinct line
styles and an optional legend. As in the archived plot, a stratum with a finite
first estimate at positive time starts at (0, 0). Nonfinite values become gaps
for every stratum; they are not replaced by zero or joined across. This corrects
the source plot's inconsistent filtering of infinite estimates in its first and
subsequent strata. Strata without finite estimates are omitted, and an entirely
nonfinite or empty result raises an explicit error. Piecewise nonfinite bins are
likewise gaps. Newly created axes use a zero hazard baseline, with a valid range
for all-zero estimates.

Plot tests verify all kernel fit types, exact curve/bin coordinates, overlay
preservation, gaps, stratified legends, empty/nonfinite rejection and all-zero
limits. A rendered three-panel figure containing global/local overlays,
piecewise bins and two-stratum failure-interval hazards was visually inspected.



## Independent time bounds and diagnostic plots

For the three bandwidth selectors, either endpoint in `bounds` can be None.
`bounds=(1, None)` uses lower time 1 and the ten-at-risk default upper time;
`bounds=(None, 5)` uses lower time zero and upper time 5, clamped to maximum
follow-up. `bounds=None` and `(None, None)` use both defaults. These settings
match the archived independently optional min/max arguments.

The help's bandwidth-function and candidate-score diagnostics use the retained
arrays. For example, after fitting `local_fit`, `global_fit` and `neighbor_fit`:

```python
from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 3, layout="constrained")
axes[0].plot(local_fit.diagnostics.time, local_fit.local_bandwidth, label="Selected")
axes[0].plot(local_fit.time, local_fit.bandwidth, label="Smoothed")
axes[0].set(xlabel="Follow-up time", ylabel="Bandwidth")
axes[0].legend()
axes[1].plot(global_fit.bandwidths, global_fit.scores)
axes[1].set(xlabel="Bandwidth", ylabel="Summed grid MSE")
axes[2].plot(neighbor_fit.neighbor_bandwidths.neighbors, neighbor_fit.scores)
axes[2].set(xlabel="Neighbor count", ylabel="Summed grid MSE")
```

For the neighbor bandwidth-function plot, replace the first panel's selected
curve with `neighbor_fit.neighbor_bandwidths.time` and
`neighbor_fit.neighbor_bandwidths.bandwidth[neighbor_fit.selected_index]`, and
use `neighbor_fit.time`/`neighbor_fit.bandwidth` for the smoothed curve. MSE score
plots require multiple candidates; constant-bandwidth bypasses do not compute
local bandwidth diagnostics. Candidate scores can span many orders of magnitude;
choose appropriate axis scales when examining the minimum. Rerun with a refined
candidate grid/count range or a different smoothing bandwidth as needed.
