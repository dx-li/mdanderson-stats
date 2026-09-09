# Legacy DCDFLIB normal distribution

`cdfnor` and `cumnor` implement the archived DCDFLIB 1.1 C/F77 normal
interfaces, with separate reference evidence from the F95 normal port.

```python
from mdanderson_stats import cdfnor, cumnor

p, q = cumnor(2)
x = cdfnor(2, p=p, q=q, mean=-3, sd=4).x
mean = cdfnor(3, p=p, q=q, x=5, sd=4).mean
sd = cdfnor(4, p=p, q=q, x=5, mean=-3).sd
```

## Interface and domains

| which | Computed group | Input parameters |
|---|---|---|
| 1 | p/q | x, mean, sd |
| 2 | x | p/q, mean, sd |
| 3 | mean | p/q, x, sd |
| 4 | sd | p/q, x, mean |

Omit the computed group. Unspecified input mean and sd default to zero and one,
a Python convenience. `DCDFLIBNormal` holds which and owned immutable broadcast
arrays p, q, x, mean, sd. Empty batches work in every mode. `cumnor(x)` returns
paired standard-normal tails. The earlier standalone CUMNOR port's `normal_tails`
API, including logarithmic probabilities, remains available.

Locations may be any finite floats and scales any positive finite floats.
**There are no legacy location or scale search bounds**, even for computed
answers: the native routine uses algebra rather than a bounded search. This
contrasts with F95 `cdf_normal`, whose location bounds remain ±1e100 and whose
SD domain remains [1e-10,1e100]. The F95 interface is unchanged.

Inversion requires positive p and q in (0,1], summing to one within three machine
epsilons. Python permits either probability to be omitted, computes its complement,
and preserves the smaller supplied tail. Zero tails are invalid for inversion;
finite forward inputs may yield zero tails through underflow. NaN/infinite input
locations and scales are rejected. A median probability cannot identify a unique
positive SD, and a computed nonpositive/nonfinite SD is invalid.

Input/domain errors and unrepresentable computed parameters raise `ValueError`.
Kernel failure or a computed answer that fails forward verification raises
`ArithmeticError`. These exceptions replace native status/bound output arguments.
The returned inverse probabilities describe the requested distribution, not an
unverified approximation at an unresolvable rounded location.

## Numerical implementation

Standard-normal tails reuse the package's compiled SciPy kernels. When a direct
tail underflows to zero, its log probability is exponentiated to recover a
representable subnormal. For example, the legacy native routine returns zero at
x=-38, while Python retains a tail of approximately 2.8854e-316. Probabilities
below the smallest representable float still round to zero.

The inverse standard-normal kernel evaluates the smaller probability. Location
and SD calculations are vectorized. For `(x-mean)/sd`, an overflowing subtraction
is replaced by `x/sd-mean/sd`; opposite-signed locations make that fallback safe
from cancellation between infinities. An overflowing affine calculation uses
half-scaled terms to recover representable cancellation before doubling. Ordinary
rows retain the original arithmetic. This recovers all four modes at x=1e308,
mean=-1e308, sd=1e308, whose standardized value is two.

Every inverse is checked by recomputing the smaller probability, with relative
tolerance 1e-7 and 32 minimum subnormals of absolute allowance. A request such as
p=0.9, mean=1e300, sd=1 has no sufficiently resolved floating-point location and
raises an error rather than returning the mean as a valid quantile.

## Validation and provenance

`tools/reference_dcdflib_normal.py` compiles the unchanged C implementation,
machine constants and header, and all 64 unchanged F77 source files in separate
executables. Fixtures retain the archive hash, hashes of compiled sources,
compiler versions, commands, drivers and empty adaptation lists. Original sources
and executables remain outside the distributed package.

Each language contributes 243 ordinary cases: 63 tails, 63 location inversions,
63 mean inversions and 54 SD inversions. All report native success. The grid spans
standardized coordinates -8, -2, -0.1, 0, 0.1, 2, 8, three means and three scales.
Five invalid-input cases check native statuses. Four additional SD cases show
success despite invalid or unidentified answers: the native median inverse is
slightly nonzero, yielding an arbitrary SD around 1.50648e16 for p=q=0.5, x=1,
mean=0. Other cases yield zero or negative SD. Python rejects all four.

Six additional cases per language retain wider-domain behavior. At x=1e308,
mean=-1e308, sd=1e308, C returns p=1/q=0 and Fortran returns NaNs, both with
status zero. The corresponding native x, mean and SD inversions overflow even
though their correct answers are representable. Python's scaled arithmetic
repairs these cases. Native underflow at -38 and a valid 1e-200 scale are also
recorded.

Independent checks use the standard library's erfc identity for both tails and
all inverse modes, with standardized coordinates through ±38 and scales from
1e-200 to 1e200. Tests also cover subnormal scales/tails, overflow cancellation,
invalid SD, the strict probability-pair contract, unresolved locations, F95
overlap, defaults, broadcasting, ownership and empty inputs.

[Batch timings](dcdflib-normal-benchmark.json) compare one vectorized call against
repeated scalar calls to the same Python API, checking matching answers. They do
not measure speedup against native C or Fortran. On the recorded machine,
2,048-element batches were about 444 times faster for tails and 615–657 times
faster for the three inversions (median of three runs).

These two legacy distribution interfaces are now covered. The remaining legacy
and public supporting interfaces keep the CDFLIB90 catalog entry partial.
