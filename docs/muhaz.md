# MUHAZ kernel hazard estimation

Catalog entry 49 is **partial**. The fixed-bandwidth kernel calculation is
implemented, together with piecewise-exponential estimates and their numerical
reports, and Nelson/product-limit failure-interval estimates. Automatic
global/local/nearest-neighbor bandwidth selection, kernel summaries and plots
remain pending. Candidate-bandwidth bias, variance and MSE diagnostics are implemented.

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
The data object retains these distinctions. Plotting will be added with the
remaining MUHAZ display workflows.

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
and storage for M distinct failure times, independently of q. Plotting remains
part of the pending MUHAZ display work.

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
bandwidth-by-time-by-quadrature-by-subject allocation. Automatic minimization,
local bandwidth smoothing and nearest-neighbor selection remain pending.
