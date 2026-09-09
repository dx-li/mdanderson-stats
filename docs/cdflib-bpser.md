# CDFLIB beta power series

`bpser(a,b,x,eps=5e-15)` computes the regularized incomplete beta integral I_x(a,b).
It preserves the source domain: positive finite shapes and tolerance, x in [0,1],
and either b<=1 or b*x<=0.7. Inputs broadcast to owned, immutable float64 arrays.

```python
from mdanderson_stats import bpser

ordinary = bpser(2, 8, 0.05)
near_one = bpser(1, 0.5, 1 - 1e-12)
subnormal = bpser(1, 0.5, 5e-324)
large_companion = bpser(100, 1e308, 5e-309)
```

## Evaluation and bounds

The shared beta-integral series sums
`1 + sum((1-b)_n*x**n*a/(n!*(a+n)), n>=1)` and combines it with the
complete normalization x**a/(a*Beta(a,b)). The normalization retains its scale
until the final multiplication, preserving representable subnormal results.

For x<=0.5, or b>1 with b*x<=0.7, future absolute term ratios after term n are
bounded by r=max(x,b*x/(n+1)). The remaining absolute sum is therefore at most
abs(term_n)*r/(1-r). This covers signed terms as well as the positive series used
by fpser and apser. Convergence is checked against the complete sum and has a
128-term work limit. The requested tolerance is capped at 5e-15 and floored at
four machine epsilons; loose tolerances do not discard the normalization or
integral correction. Nonconvergence raises `ArithmeticError`.

For b<=1 and x>0.5, the existing legacy beta adapter avoids the slow series near
one. The native routine timed out at the valid inputs (a=1,b=0.5,x=1) and
(a=1,b=0.5,x=1-1e-12); this port handles endpoints directly and uses stable unit-shape
identities. At the exact half-subnormal tie with a=1 and b<1, an exact integer-ratio
check retains the positive quadratic correction that determines rounding.
Thus `bpser(1,0.5,5e-324)` returns the smallest positive float64, not zero.

For b>=1e15 and a*a/b<=1e-14 (tested in logarithms), the large-companion gamma
approximation P_gamma(a,b*x) avoids cancellation between huge normalization
logarithms. The leading relative correction in this small-product domain is of
order (a*a+a)/b; the separate lower bound on b controls the linear term when a<1.
This is an approximation, independently checked against beta integrals and across
its transition. General incomplete-gamma leading terms for large-parameter beta
expansions are described in [NIST DLMF 8.18](https://dlmf.nist.gov/8.18#ii).

The product-domain check uses exact integer ratios when floating-point division
lands on the boundary, so rounded multiplication cannot admit an invalid input.
Invalid input raises `ValueError`; nonfinite or out-of-range numerical output
raises `ArithmeticError`. True underflow is allowed. eps bounds series truncation,
not every kernel or approximation error, and is not a correct-rounding guarantee.

## Validation and performance

[Tests](../tests/test_cdflib_bpser.py) compare native audit cases and a broad grid
against independent 800-digit beta integrals. Cases include signed series above
x=0.5, smallest-positive and enormous shapes, near-one coordinates, loose tolerances,
exact domain boundaries, gamma-limit transitions, broadcasting and ownership.
Existing fpser and apser tests cover the shared remainder-bound change.

[Benchmarks](cdflib-bpser-benchmark.json) compare one batched call against repeated
scalar calls, requiring identical results. These measure batching benefits rather
than speed relative to Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
