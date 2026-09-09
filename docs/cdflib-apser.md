# CDFLIB small-first-shape upper beta tail

`apser(a,b,x,eps=5e-15)` computes I_(1-x)(b,a), the upper beta tail.
It retains the source domain: positive finite a, b and eps;
`a <= min(eps,eps*b)`; `b*x <= 1`; and `0 <= x <= 0.5`.
Inputs broadcast to independently owned, immutable float64 results.

```python
from mdanderson_stats import apser

ordinary = apser(1e-20, 0.5, 0.5)
endpoint = apser(1e-20, 1, 0)
small_companion = apser(5e-324, 1e-309, 0.5)
large_companion = apser(1e-309, 1e308, 5e-309)
loose_tolerance = apser(0.01, 0.5, 0.1, eps=0.1)
```

## Domain and evaluation

Shape and coordinate-product comparisons avoid overflowing or underflowing
products. A quotient rounded onto the comparison boundary is resolved using
exact integer ratios of the binary floating-point inputs. The shared validator
supports the inclusive apser inequalities and the strict fpser inequality.
Consequently, a rounded product of one does not conceal an exact product above
one. Invalid domains raise `ValueError`.

At x=0 the upper tail is one, repairing the native infinite return. Unit shapes
use exact log1p/expm1 identities.

When a+b<=0.25, a gamma recurrence separates b/(a+b) from a correction factored
in a*b. The positive beta-integral series provides the remaining term. Keeping
these pieces in log(P) and using -expm1(log(P)) retains Q when P rounds to one.
This also repairs the native overflow at a=the smallest positive float64 and
b=1e-309, where Q is finite and approximately 4.9406564584e-15.

The integral series is shared with fpser. Its divided form preserves the
small-first-shape coefficient before multiplication by a. A geometric remainder
bound applies for x<=0.5; the loop has a limit of 128 terms. The requested series
tolerance is capped at 5e-15 and floored at four machine epsilons.

For remaining cases with a<=1e-18 and b<1e15, the first-order small-a expansion
uses a 64-term series. Here b is bounded away from zero by the earlier
small-shapes branch. Under x<=0.5 and b*x<=1, successive absolute series terms
shrink by at least a factor of two. The very small a threshold also limits the
omitted higher-order shape corrections.

At b>=1e15, the implementation uses the large-companion approximation Q_gamma(a,b*x)
through the existing stable incomplete-gamma helper. Incomplete-gamma leading
terms in large-parameter beta expansions are described in
[NIST DLMF 8.18](https://dlmf.nist.gov/8.18#ii). This path is independently
checked against beta integrals, including tiny upper tails and loose tolerances.

Other cases use the shared [beta-tail calculation](cdflib90.md#numerics-and-failure-behavior).
Its log-domain path for very small coordinates also handles the subnormal cases
originally repaired inside apser. The shared path benefits distribution interfaces
and avoids maintaining a duplicate normalization calculation.

The tolerance controls series truncation and the allowed source domain; it is
not a guarantee of correct rounding or a bound on every approximation/kernel
error. True underflow is allowed. Nonconvergence or invalid numerical output
raises `ArithmeticError`.

## Validation and performance

[Tests](../tests/test_cdflib_apser.py) compare the native audit cases and a broader
shape/coordinate grid against independent 800-digit beta integrals. They include
very small and huge shapes, an overflowing shape sum, loose tolerances, exact
rounded domain boundaries, numerical regime transitions, broadcasting and
immutable ownership. The fpser tests also validate the shared-series and
validator changes.

[Benchmarks](cdflib-apser-benchmark.json) compare batched calls with repeated
scalar calls to this Python API and require identical results. They measure
batching benefits, not speed relative to Fortran.

The later [bpser](cdflib-bpser.md) and [bgrat](cdflib-bgrat.md) ports bring coverage
to 32 of the 35 F95 mathematical procedures. `basym`, `bfrac` and `bratio` remain, along with other CDFLIB support interfaces
and the rest of the software catalog. CDFLIB90 remains partial.
