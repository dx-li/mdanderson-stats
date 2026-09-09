# CDFLIB tiny-companion beta series

`fpser(a,b,x,eps=5e-15)` computes the regularized beta integral I_x(a,b).
It preserves the source domain: finite positive shapes and tolerance,
`b < min(eps,eps*a)`, and `0 <= x <= 0.5`. Inputs broadcast through NumPy;
results are independently owned, immutable float64 arrays.

```python
from mdanderson_stats import fpser

ordinary = fpser(0.5, 1e-20, 0.5)
endpoint = fpser(1e-20, 1e-40, 0)
subnormal = fpser(1, 1e-15, 1e-308)
loose_tolerance = fpser(0.5, 0.2, 0.5, eps=2)
```

## Domain and evaluation

The strict shape inequality is tested through b/a, avoiding a product that
could overflow or underflow. When division rounds exactly onto eps, integer
ratios of the binary floating-point inputs resolve the strict comparison.
An invalid domain raises `ValueError`, including nonpositive eps.

At x=0 the result is zero, including tiny a values for which the source skips
its power calculation. Unit shapes use exact exponential identities with
log1p/expm1. The audited lost probability of 1e-323 is retained.

For b<1, the defining beta-density power series has positive integral terms.
Each successive term has ratio at most x, so a geometric bound controls the
uncomputed remainder. The loop is capped at 128 terms. Its tolerance is capped
at 5e-15 for loose requests and floored at four machine epsilons. Failure to
satisfy the bound raises `ArithmeticError`.

The implementation retains the complete beta normalization, using the existing
stable beta factors and normalization helper. It removes the complementary
coordinate power and combines the remaining scale with the series before the
final result is formed. It does not discard power factors at the smallest
normal floating-point value. This repairs the native subnormal loss and avoids
using the approximation 1/Beta(a,b) = b for loosely specified tolerances.

The literal source inequality can admit b>1 when eps is very large. Such inputs
use the existing compiled beta-tail kernel, keeping the stated input domain
without applying a tiny-b approximation to a large companion shape. A kernel
failure or invalid numerical result is reported explicitly.

The tolerance controls series truncation, not a guarantee of correct rounding
or a bound on every floating-point/kernel error. True underflow is allowed.

## Validation and performance

[Tests](../tests/test_cdflib_fpser.py) compare all audited fpser calls with
independent high-precision beta integrals. They cover source failures, strict
rounded boundaries, positive-shape extremes, loose tolerances, subnormal
results, broadcasting and immutable ownership. Non-unit-shape subnormal cases
exercise the general series path rather than the exact unit-shape identity.

[Benchmarks](cdflib-fpser-benchmark.json) compare batches of 64 and 256 coordinates
with repeated scalar calls to this Python API, requiring identical results.
They measure batching benefits, not speed relative to Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Imported constants and other CDFLIB support interfaces
remain open, along with the rest of the software catalog. CDFLIB90 remains partial.
