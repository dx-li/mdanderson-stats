# CDFLIB accumulated beta increment

`bgrat(a,b,x,y=None,w=0,eps=5e-15)` returns w+I_x(a,b), preserving the source
operation of adding a beta integral to an existing accumulator. The source
shape domain is a>=15 and 0<b<=1. Shapes, accumulator and tolerance must be
finite, and eps must be positive. Inputs broadcast to independently owned,
immutable float64 results; supplied arrays are not mutated.

```python
from mdanderson_stats import bgrat

increment = bgrat(15, 0.5, 0.5)
updated = bgrat(15, 0.5, 0.5, w=-1)
endpoint = bgrat(15, 0.5, 1, w=0.25)  # 1.25
extreme = bgrat(1e308, 0.5, None, y=5e-324)
```

## Evaluation and source repairs

Coordinates x and y are complementary probabilities. Supply either or both;
when supplying only y, pass x=None. The common pair validator preserves the
smaller supplied coordinate and reconstructs the larger one, accepting a pair
sum within eight machine epsilons. An explicit tiny y remains meaningful even
when x rounds to one.

For the smaller x coordinate, the implementation uses the validated bpser beta
integral. Its bounded series retains the full normalization, with truncation
tolerance capped at 5e-15 and floored at four machine epsilons. For the larger x
coordinate, the existing legacy beta adapter evaluates the integral using both
coordinates, preserving the complementary-coordinate information. The accumulator
is added only after evaluating the increment. It may be negative or exceed one;
the result is not clipped to a probability range.

The [native audit](cdflib-beta-remaining-reference.md) recorded NaNs at x=0,
no increment with ierr=1 at x=1, and failures at enormous first shapes where the
increment provably underflows. This port returns the correct endpoint increments
0 and 1 and a zero increment on true underflow. Invalid inputs raise `ValueError`;
numerical failures propagate as `ArithmeticError`, following the package's
exception conventions instead of returning a native ierr code.

The implementation reuses the complete beta-integral calculations. eps controls
the bounded series when used; it is not a bound on every kernel error or a guarantee of
correct rounding.

## Validation and performance

[Tests](../tests/test_cdflib_bgrat.py) compare all native bgrat audit cases and a
broader shape/complement grid against independent 800-digit integrals and proven
underflow bounds. They cover smallest-positive companion shapes and complements,
a up to 1e308, exact endpoint increments, signed and large accumulators, invalid
inputs, broadcasting and immutable ownership. Unit companion shapes independently
check the exact increment x**a.

[Benchmarks](cdflib-bgrat-benchmark.json) compare batched calls with repeated
scalar calls and require identical outputs. They measure batching benefits,
not speed relative to Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
Other CDFLIB support interfaces and the rest of the catalog remain open.
CDFLIB90 remains partial.
