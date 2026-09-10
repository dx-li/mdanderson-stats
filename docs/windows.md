# WINDOWS smoothing

Catalog entry 61 provides moving-window estimators and cross-validation. This
increment covers all **fixed-width** estimators, with boxcar and biquadratic
weights. Nearest-neighbor windows, their boundary-tie weights, and full native
workflow coverage remain pending; the catalog status is partial.

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
explicitly before calling. A width is required rather than silently chosen.

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
endpoint extrapolation. Native nearest-neighbor parity is not claimed.

Sources: [MD Anderson WINDOWS](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/61)
and [original archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/WINDOWS/WINDOWS_V1.tar.gz).
Hashes of the archive and consulted sources are in `windows-sources.json`.
Original files remain in ignored research storage and are not redistributed.
