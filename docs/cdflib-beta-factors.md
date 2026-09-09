# CDFLIB beta scaling factors

`brcomp(a,b,x,y=None)` computes x^a y^b/B(a,b).
`brcmp1(mu,a,b,x,y=None)` includes the factor exp(mu). Inputs broadcast through
NumPy; results are independently owned, immutable float64 arrays.

```python
from mdanderson_stats import brcomp, brcmp1

ordinary = brcomp(2, 3, 0.1)
huge_center = brcomp(1e308, 1e308, 0.5, 0.5)
scaled_tiny_shapes = brcmp1(1000, 1e-309, 1e-309, 0.5, 0.5)
scaled_huge_shapes = brcmp1(-1000, 1e300, 1e300, 0.5, 0.5)
negative_shape = brcomp(-0.5, 1, 0.5, 0.5)
```

## Input and result contracts

At least one coordinate must be provided. Coordinates must be in [0,1], and
when both are supplied they must sum to one within eight machine epsilons. The
existing CDFLIB pair validator preserves the smaller supplied probability and
reconstructs the larger one. This retains tiny tails whose complements round to
one. Endpoint coordinates return zero, preserving the native endpoint behavior.

`mu` accepts signed int32 values. All other inputs must be finite. Positive
shapes are supported even when their sum overflows float64. Negative noninteger
shapes are supported where Gamma(a), Gamma(b) and Gamma(a+b) are defined on the
real axis; their signs are retained. Gamma poles raise `ValueError`. A zero shape
with a positive companion preserves the source's zero return.

Output overflow raises `ArithmeticError`. True underflow is allowed. The
interfaces do not promise correctly rounded values at every input; relative
accuracy can deteriorate when a large scale cancels a large log factor or when
extreme negative shapes require differences of log-gamma values.

## Stable evaluation

For ordinary positive shapes, gamma recurrences separate a small multiplicative
shape factor from the remaining exponent. Tiny shapes reuse the factored local
log-gamma series; disparate shapes reuse the stable gamma-ratio helper. Unit
shapes use their exact beta identity.

Exponential scaling includes the complete scale before evaluating the result.
Normal exponent ranges use one final multiplication, wider ranges split the
exponential, and extreme ranges combine the prefactor logarithm too. This repairs
the audit's false infinities, lost finite results and premature subnormal
rounding. Unneeded numerical branches are skipped for each batch.

For two large shapes, a Stirling representation avoids the overflowing shape
sum. The deviation from the beta center uses compensated products for a*y−b*x.
The product calculation splits normalized mantissas, so the compensation step
itself cannot overflow. A separate correction accounts for the rounding of the
larger complementary coordinate. These details preserve meaningful deviations
near the center at shapes such as 1e30.

Stable log1p remainders evaluate the center correction. In the far tails, normal
coordinate ratios are formed before taking a logarithm, avoiding cancellation
between large logarithms for extreme companion shapes. Subnormal ratios retain
separate logarithms to avoid quantizing the coordinate. Direct 800-digit factor
checks cover both orientations of these extreme-companion cases. The existing
gamma/Stirling helpers are reused.

On the negative-shape domain, a compensated sum retains the small offset when
a+b rounds onto or close to a gamma pole. Reflection evaluates the gamma factor
using that offset, so domain validation rejects true poles and preserves nearby
valid inputs.

## Validation and performance

[Tests](../tests/test_cdflib_beta_factors.py) cover every applicable native factor
case against independent high-precision values, scaled extremes, near-center
offsets, subnormal coordinates, negative shapes, poles, broadcasting and
immutability. They include a negative shape with a 1e308 companion and a scaled
subnormal coordinate with mu=6000. The [native audit](cdflib-beta-factors-reference.md)
retains source provenance and distinguishes native defects from valid behavior.

[Benchmarks](cdflib-beta-factors-benchmark.json) compare batches of 64 and 256
coordinates against repeated scalar calls to the same API, requiring identical
results. Cases include ordinary, tiny, large-center, scaled and negative-shape
factors. These measure batching benefits, not a speedup over Fortran.

The later [beta shape-shift port](cdflib-beta-shift.md) implements `bup`.
Seven remaining incomplete-beta helpers, constants and other support interfaces
keep CDFLIB90 partial. The full catalog conversion remains ongoing.
