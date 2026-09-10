# WINDOWS smoothing

Catalog entry 61 provides moving-window estimators and cross-validation. The
implementation covers all ten estimators under both fixed-width and
nearest-neighbor membership, with boxcar and biquadratic weights, and
cross-validation for mean/linear/quadratic fits. The original terminal prompts
and text plotting are represented by array-based Python inputs and outputs.

```python
import numpy as np
from mdanderson_stats import window_smooth, window_cross_validation

x = np.arange(-4.0, 5.0)
y = 2 + 3 * x - 0.5 * x * x
fit = window_smooth(x, y, width=6, centers=[-2, 0, 2], estimator="quadratic")
assert np.allclose(fit.estimates, [-6, 2, 6])
cv = window_cross_validation(x, y, [4, 6, 8], estimator="quadratic")
assert cv.best_width is not None
```

`width` is the **full** width: membership includes both endpoints of
`center ± width/2`. Input observations are stably sorted by x. Omitted centers
produce 20 equally spaced values over the observed range; explicit centers keep
their input order. Inputs must be finite; missing observations must be handled
explicitly before calling. Supply exactly one of `width` or `neighbors` rather than silently choosing a smoothing parameter.

Available `estimator` values are `mean`, `linear`, `quadratic`, `maximum`,
`minimum`, `quantile`, `std`, `derivative`, `span`, and `count`.
`derivative_degree=1` or `2` selects the polynomial used to estimate the **first**
derivative. `span` reports the observed x range inside the window, not its
requested width. `count` includes every observation, including repeated x values.
Unlike the native routine, count and span also work for singleton windows; an
empty count is zero. Other empty estimates are NaN with status `empty`.
Insufficient observations and polynomial rank loss have distinct statuses.
A numerical overflow raises an error rather than returning an invalid estimate.

`weighting="boxcar"` gives equal weights. `"biquadratic"` uses normalized
`(1 - ((x - center)/width)**2)**2`. The denominator is the full width, as in the
source; this differs from kernels that fall to zero at the window endpoints.
The `std` estimator is the weighted population standard deviation, without a
sample degrees-of-freedom correction. Minima, maxima, counts and spans ignore
weights. Returned centers, estimates, counts and spans are read-only arrays.

## Polynomial and quantile conventions

By default, local polynomial least squares uses the kernel weights. The native
`lqbeta` routine instead uses their reciprocals; select
`polynomial_weights="native_inverse"` to reproduce that convention. Equal-weight
fits coincide. Polynomial fitting uses centered/scaled local coordinates and SVD,
avoiding the original normal-equation inverse and global powers of x.

Weighted quantiles sort y, place each value at its cumulative-weight midpoint,
and linearly interpolate. Values below the first midpoint or above the last use
linear extrapolation from the nearest pair, so quantiles at zero or one may lie
outside the observed y range. At least two observations are required. Python
corrects the original upper-tail branch, which references an unsorted y value
and accumulates the final full weight rather than the final midpoint. Quantiles
outside [0,1] are rejected; the S wrapper's impossible conjunction is corrected.

## Cross-validation

`window_cross_validation` evaluates explicit positive width candidates with
`estimator="mean"`, `"linear"` or `"quadratic"`. Each fold removes exactly the
observation being predicted, retaining other observations at the same x.
Membership stays centered on the held-out observation. The same calculation is
available as `window_smooth(..., leave_one_out=True)`; estimates then follow sorted
observation order and explicit centers are disallowed.

A candidate receives SSE only if **all** folds are valid. Otherwise its SSE is
NaN, and `valid_folds` reports how many predictions were estimable. `best_width`
is the first minimizing valid candidate, or `None` if none is valid. This avoids
the original wrapper's possible use of uninitialized predictions for empty
windows. Candidate grids are explicit, avoiding its default upper endpoint's
dependence on max(x) rather than the x range.

Membership bounds use binary searches in sorted x, with work proportional to
selected observations rather than a dense center-by-observation distance matrix.
Cross-validation applies storage/work limits before large calculations.

## Validation and sources

The original `S/windows/windowing.f` was compiled with gfortran and Accelerate
BLAS. All ten fixed-width estimators under both weighting schemes were compared
at three centers, with inverse weighting enabled for native polynomial parity.
They agree within the original single-precision tolerance (relative 1e-5,
absolute 2e-6). Native mean and linear leave-one-out predictions also agree.
`tests/fixtures/windows-native.json` stores the estimator outputs for synthetic
x=0,...,6 and y=(2,0,4,1,8,3,7), width 4 and centers (0,2.5,5).
Further focused checks cover exact quadratic reproduction, first derivatives,
rank loss, empty windows, repeated-x leave-one-out behavior and corrected quantile
endpoint extrapolation. Nearest-neighbor validation is described below.

Sources: [MD Anderson WINDOWS](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/61)
and [original archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WINDOWS/WINDOWS_V1.tar.gz).
Hashes of the archive and consulted sources are in `windows-sources.json`.
Original files remain in ignored research storage and are not redistributed.

## Nearest-neighbor windows

```python
from mdanderson_stats import window_neighbor_cross_validation, window_smooth

nearest = window_smooth([0, 1, 1.1, 100], [0, 1, 2, 100], neighbors=2, centers=[1.1])
assert abs(nearest.estimates[0] - 1.5) < 1e-12
neighbor_cv = window_neighbor_cross_validation(np.arange(9), [2, 0, 4, 1, 8, 3, 7, 2, 9], [3, 5, 7])
assert neighbor_cv.best_neighbors in (3, 5, 7)
```

`neighbors` is an integer from 2 through the number of input observations.
A binary search selects the closest block of k observations, and all observations
at the kth distance are included. Thus the reported count may exceed k. Ties
use exact floating-point distances and x values, avoiding the source's
coordinate-dependent approximate-equality rule. This makes membership invariant
to observation ordering, including duplicate x values. As with any floating-point
distance calculation, very large coordinate offsets can make distinct distances
numerically indistinguishable; inconsistent boundary mass raises an error.

For n selected points, let b be the number belonging to either endpoint-value
group. Boundary weights receive factor `(k - (n - b))/b`, with unit weights on
interior points, then normalization. This reproduces the original boxcar boundary
adjustment. For biquadratic weighting the denominator is the selected window's
**observed span**, and the boundary factor is applied when b>2, as in the source.
The polynomial kernel is evaluated as written even for centers outside the data
range; it is not replaced by a compact-support kernel. Its weights are evaluated
in the log domain to avoid overflow far from the data range.

If all selected x values coincide with the center, kernel weights are uniform.
A zero-span biquadratic window away from its center has status `zero_weight` for
weighted estimators. Zero kernel weights are removed before estimation; native
inverse-weight polynomial fits additionally omit normalized weights at most
1e-10, matching `lqbeta`. Counts and spans still describe the selected membership.
Min/max/count/span ignore weights. Insufficient positive-weight polynomial
observations retain the existing explicit failure statuses.

`window_neighbor_cross_validation` tests explicit integer neighbor counts.
Selection, span and boundary weights are computed **before** dropping the target
observation, as in WINDOWS. Only its row is excluded, and remaining weights are
renormalized. Consequently k usually supplies k−1 training observations, with
additional observations possible from ties. The result exposes `neighbors`,
`sum_squared_errors`, `valid_folds`, and `best_neighbors`. Invalid candidates
follow the same rules as width cross-validation; repeated x values are retained.

### Native corrections and validation

The original `wnnrn` starts with the closest point and its right neighbor, then
compares included boundary distances instead of the next candidate distances.
On x=(0,1,1.1,100), center=1.1 and k=2 it selects (1.1,100), despite 1 being
closer. Python selects (1,1.1). It also includes both equal-distance boundaries
when k would otherwise split a tie. The source has a double increment in its
right-boundary tie scan and possible out-of-range accesses; Python replaces
that scan with bounded searches and does not emulate those defects.

`windows-neighbor-native.json` covers all ten estimators and both weighting
schemes on x=0,...,8, y=(2,0,4,1,8,3,7,2,9), k=5, and centers (2.5,4.5,5.5).
The original Fortran and Python agree at relative 1e-5/absolute 2e-6 tolerance
with native inverse polynomial weights. The two native weighting routines were
also checked directly on x=(0,0,1,2,2), k=4, center=1, independently of membership.
Exhaustive-distance checks cover irregular designs, extrapolation and k=n.
Independent R fits supply all 108 leave-one-out predictions for mean, linear
and quadratic fits, both kernels and both polynomial-weight conventions on the
nine-observation data with k=5. These agree to 1e-12 and are stored in
`windows-neighbor-cv.json`. Tests also cover zero weights, all-equal x values,
quadratic reproduction and invalid cross-validation candidates.

The archived introduction's implemented estimator list is covered. The manual
also discusses possible applications such as local Kaplan–Meier estimation and
coefficient plots; these are not routines implemented in the distributed WINDOWS
estimator dispatch and are not claimed as WINDOWS features here.
