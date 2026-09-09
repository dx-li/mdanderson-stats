# CDFLIB gamma-ratio foundations

Three more public F95 procedures now have vectorized implementations. They form
the numerical foundation for the beta/combinatorial helpers.

| Function | Result | Domain |
|---|---|---|
| `algdiv(a,b)` | log Γ(b)−log Γ(a+b) | Finite a,b; b≥8 and a+b>0 |
| `bcorr(a,b)` | δ(a)+δ(b)−δ(a+b) | Finite a,b≥8 |
| `gsumln(a,b)` | log Γ(a+b) | Finite a,b in [1,2] |

δ denotes the Stirling correction in
log Γ(x)=(x−1/2)log(x)−x+log(2π)/2+δ(x).

Arguments broadcast and results are owned immutable float64 arrays. Scalar-shaped
and empty results are supported. Invalid domains raise `ValueError`; a log ratio
outside the finite output range raises `ArithmeticError`. A positive input sum
need not fit in float64, provided the result does.

```python
from mdanderson_stats import algdiv, bcorr, gsumln

ratios = algdiv([1e-100, -9.5], [1e308, 10])
small_correction = bcorr(1e308, 1e308)
close_sum = gsumln(1, 1 + 2**-52)
```

## Numerical approach

The Stirling correction uses ten exact Bernoulli coefficients evaluated with
Horner's rule in 1/x². For x≥8, the first omitted term is about 1.5e-18 or less.
`bcorr` forms the reciprocal of a+b as `(1/max)/(1+min/max)`, so an overflowing
sum cannot destroy the correction. It evaluates the corrections directly rather
than subtracting large log-gamma values.

For nonnegative a, `algdiv` separates its leading logarithmic terms and a stable
Stirling-correction difference. It reassociates
(b+a−1/2)log(1+a/b) as a times a factor that tends to 1−1/(2b) when a/b tends to
zero. Thus even a quotient rounded to zero does not erase the leading contribution.
The correction difference uses finite geometric sums instead of subtracting
nearly equal corrections.

For negative a, the ratio is reversed and both gamma arguments are shifted into
the region where the Stirling series is accurate. At most eight recurrence terms
are needed. Close recurrence factors use log1p; well-separated factors use their
individual logarithms to avoid rounding a ratio to −1. This preserves the valid
negative domain and repairs the native (-8,10) and (-9.5,10) failures.

`gsumln` first computes `(a−1)+(b−1)`, preserving tiny offsets from two. It then
uses log-gamma recurrence with the already validated local remainder function.
It does not form a rounded a+b before extracting the offset.

## Validation and performance

The [native audit](cdflib-beta-support-reference.md) provides unchanged F95
reference cases. Implementation checks compare against independent high-precision
identities and cover negative shifts, quotient underflow, overflowing positive
sums, subnormal corrections, branch boundaries, broadcasting and ownership.

The independent ratio oracle uses a digamma derivative for relative increments
below 1e-30. Its next relative term is O(|a|/b). This avoids subtracting the errors
of two independently truncated Stirling expansions when the desired difference
is much smaller. Tiny positive and negative increments are also checked against
the exact harmonic-number identity for digamma at eight. Tests therefore verify
small results without treating an oracle's absolute approximation error as truth.

[Benchmarks](cdflib-gamma-ratios-benchmark.json) compare 64- and 256-value batches
with repeated scalar calls to the same API, checking identical results. They
include negative shifts, subnormal ratios and corrections. These measure batching
benefits, not a native-language speedup. No dependency or existing distribution
kernel was changed. As with ordinary float64 log-gamma evaluation, relative
accuracy can deteriorate near nontrivial zero log ratios on the negative-a domain.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Imported constants and other CDFLIB support interfaces
remain open, along with the rest of the software catalog. CDFLIB90 remains partial.
