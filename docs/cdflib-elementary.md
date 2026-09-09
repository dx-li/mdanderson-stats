# CDFLIB elementary mathematical support

These five public F95 `biomath_mathlib_mod` procedures have vectorized Python
implementations. Subsequent ports complete all 35 mathematical procedures;
their [constants](cdflib-constants.md) are also implemented.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.

| Function | Mathematical result | Domain |
|---|---|---|
| `alnrel(a)` | log(1+a) | finite a>-1 |
| `rexp(x)` | exp(x)-1 | finite x, finite representable result |
| `rlog(x)` | x-1-log(x) | positive finite x |
| `rlog1(x)` | x-log(1+x) | finite x>-1 |
| `evaluate_polynomial(a,x)` | a[0]+a[1]x+… | nonempty finite 1-D coefficients; finite x |

```python
from mdanderson_stats import rlog1, evaluate_polynomial

small = rlog1([1e-8, 1e-100, 3e-162])
values = evaluate_polynomial([1, 2, 3], [-1, 0, 1])
```

Scalar or array coordinates preserve their shape and return owned immutable
float64 arrays, including empty batches. Polynomial coefficients are shared
across the coordinate batch and ordered from constant to highest power. Invalid
input domains raise `ValueError`. Nonfinite exponential or Horner results raise
`ArithmeticError`, replacing native infinities or undefined evaluations. Horner
intermediate overflow is an error even if a different evaluation scheme could
produce a finite final result; this port does not promise compensated polynomial
evaluation near ill-conditioned roots.

`alnrel` and `rexp` use NumPy's cancellation-resistant `log1p` and `expm1`.
For |x|<=1/8, `rlog1` evaluates the series
x²(1/2-x/3+x²/4-…), through degree 40, using Horner's rule. Its omitted relative
tail is below 2(1/8)^39. Multiplication is ordered as `(x*series)*x`, avoiding
premature underflow of x². Outside that interval, direct x-log1p(x) has adequate
separation. `rlog` uses this same remainder with the accurately represented x-1
near one and evaluates its defining expression elsewhere.

## Native evidence and independent checks

`tools/reference_cdflib_elementary.py` compiles the unchanged constants and full
mathematical module with bounds checking and records 90 native calls. The fixture
retains source/archive hashes, compiler identity, command and driver; there are
no source adaptations. This validates these F95 procedures, not separate legacy
C/F77 contracts that merely share their names.

Independent 800-digit Decimal calculations verify values near zero, near one,
at the series switch and throughout the recorded ordinary domain. Polynomial
checks establish coefficient order and the identity (x-1)^4. Additional checks
cover subnormal first-order functions, finite output limits, broadcasting,
immutability and empty coordinate batches.

At x=±3e-162, the native `rlog1` squares x/2 before multiplying by two. That square
underflows to zero, although the mathematical remainder rounds to the minimum
positive float. The Python multiplication order preserves the positive result.
The native sentinel -1 for `rlog1(x<=-1)` becomes a clear domain error. An empty
coefficient array causes a native bounds failure and is rejected before evaluation
in Python. Native exponential overflow is likewise reported explicitly.

[Benchmarks](cdflib-elementary-benchmark.json) compare one batch call with repeated
scalar calls to the same Python API at 64 and 256 coordinates, checking identical
outputs. They do not measure a native-language speedup. The implementations use
NumPy array operations and a short fixed-length polynomial loop, with no added
dependency or change to existing distribution kernels.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
