# MUHAZ kernel hazard estimation

Catalog entry 49 is **partial**. The fixed-bandwidth kernel calculation is
implemented; automatic global/local/nearest-neighbor bandwidth selection,
MSE diagnostics, piecewise-exponential and Kaplan–Meier-type estimates,
high-level summaries and plots remain pending.

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
