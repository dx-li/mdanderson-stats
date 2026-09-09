# CDFLIB beta and combinatorial logarithms

The remaining three interfaces from the beta-support audit now have vectorized
Python implementations:

| Function | Result | Domain |
|---|---|---|
| `betaln(a,b)` | log B(a,b) | Positive finite shapes |
| `log_beta(a,b)` | Renamed interface to log B(a,b) | Positive finite shapes |
| `log_bicoef(k,n)` | log Γ(n+1)−log Γ(k+1)−log Γ(n−k+1) | Finite n>−1 and −1<k<n+1 |

The combinatorial interface preserves real-valued inputs, including valid
fractional and negative values. Arguments broadcast and return owned immutable
float64 arrays, including scalar-shaped and empty results. Invalid domains raise
`ValueError`; nonfinite output raises `ArithmeticError`. Genuine underflow of a
small logarithm is allowed. This is not an integer-only counting interface.

```python
from mdanderson_stats import betaln, log_beta, log_bicoef

huge_shapes = betaln(1e308, 1e308)
close_shapes = log_beta(1, 1 + 2**-52)
fractional = log_bicoef([0.5, -0.25], [1, -0.5])
tiny_endpoint = log_bicoef(1e-100, 1)
tiny_pair = log_bicoef(1.5e-162, 3e-162)
small_total = log_bicoef(1, 5e-324)
```

## Numerical implementation

Unit beta shapes use B(1,b)=1/b directly. Small shapes use the validated log-gamma
helpers, preserving offsets from two in a+b when cancellation matters. Near two
unit shapes, the factored mixed log-gamma series also preserves the quadratic
result when the two offsets cancel. Disparate
shapes use the stable gamma ratio. When both shapes are large, the source's
Stirling-correction arrangement avoids overflowing a+b or forming huge gamma
values. The two beta names share one implementation and are exactly symmetric
under exchanging their inputs.

The combinatorial helper uses several complementary representations:

* Exact endpoints return zero, and k=1 uses log(n). The right beta shape is
  assembled in an order that preserves n near k=1 and n+1 near n=−1; valid inputs
  such as k=1,n=5e-324 are not mistaken for a zero denominator shape.
* Near an endpoint, shifted gamma ratios and log1p recurrence retain the small
  increment. For recurrence ratios far from one, separate logarithms preserve a
  tiny positive denominator shape next to the fractional domain boundary.
  Extremely small increments use the digamma derivative, multiplying
  only after the coefficient is evaluated. This avoids splitting a subnormal
  increment into quantities that round to zero.
* When |k|, |n−k| and |n| are at most 1/4, the log-gamma Taylor identity factors
  k(n−k) before evaluation. The cancelling linear terms never form. A fixed
  40-degree recurrence with zeta coefficients evaluates the remaining series;
  its truncation is below float64 resolution in this region. The larger factor
  is multiplied by the series first, leaving subnormal rounding to the final
  product.
* The regular region uses the stable beta logarithm. This retains finite
  logarithmic coefficients even for totals of 1e308.

SciPy supplies the zeta coefficients through its documented q=1 Hurwitz-zeta
interface. The implementation otherwise reuses the package's gamma/digamma and
gamma-ratio helpers with NumPy array operations. No dependency or existing
distribution kernel changed.

## Evidence and limits

The [native audit](cdflib-beta-support-reference.md) records unchanged F95 behavior.
Tests compare against independent high-precision gamma identities, exact integer
coefficients and unit-beta identities, including every applicable audit case.
Additional tests cover tiny inputs on both sides of zero, subnormal final values,
large shapes, close unit shapes, fractional domains, domain boundaries,
broadcasting and immutability. The independent combinatorial oracle also uses a
digamma derivative for a tiny endpoint increment, avoiding subtraction of larger
Stirling approximation errors.

The functions do not promise correctly rounded transcendental results at every
input. Relative accuracy can deteriorate near nontrivial zeros of log beta;
small endpoint and near-origin combinatorial values receive explicit treatment.

[Benchmarks](cdflib-beta-support-benchmark.json) compare batches of 64 and 256
coordinates with repeated scalar calls to the same API, checking identical
results. They cover wide beta shapes, close unit shapes, tiny endpoints, small
argument pairs and huge totals. These measure batching benefits, not a speedup
against Fortran.

The later [gamma scaling factor](cdflib-gamma-factor.md) adds `rcomp`. Remaining
incomplete-beta/gamma routines, constants and other support contracts keep
CDFLIB90's catalog status partial.
