# Legacy DCDFLIB gamma

`cdfgam` and `cumgam` independently implement the archived C/F77 gamma
interfaces. They are separate from the more tightly bounded F95 `cdf_gamma`.

```python
from mdanderson_stats import cdfgam, cumgam

forward = cdfgam(x=2, shape=3, rate=0.5)
x = cdfgam(2, p=forward.p, q=forward.q, shape=3, rate=0.5).x
shape = cdfgam(3, p=forward.p, q=forward.q, x=2, rate=0.5).shape
rate = cdfgam(4, p=forward.p, q=forward.q, x=2, shape=3).rate
p, q = cumgam(1, 3)  # Already-scaled, unit-rate coordinate.
```

## Contract

Modes 1–4 compute p/q, x, shape and rate respectively. Omit the computed group.
Input rate defaults to one. The source calls this parameter SCALE, but multiplies
x by it: it is a **rate**, not the conventional gamma scale. All supplied values
must be finite; x is nonnegative and shape/rate are strictly positive.
Broadcast results are owned, immutable arrays in `DCDFLIBGamma`.

The source header advertises bounds on computed x and SCALE, but the executable
uses `gaminv` followed by division without enforcing those bounds. Python therefore
allows every representable finite nonnegative x and positive rate. Only computed
shape has the executable search bounds [1e-100, 1e100]. Input shape is not capped.
The unchanged native fixtures demonstrate successful x and rate results of 1e200.

Supply at least one of p/q for inversions. A supplied pair must sum to one within
three machine epsilons; the smaller tail is preserved. Inversion requires q>0.
Quantile p=0 gives x=0. Shape/rate inversion additionally requires positive x and
p: zero-coordinate cases do not identify those parameters, and a zero inferred
rate would violate the distribution's domain. The native routine instead returns
rate=0 with status 0 for p=0, x=1, shape=2; Python rejects it.

Invalid input, unrepresentable answers and unattainable shape bounds raise
`ValueError`. Invalid numerical kernels or failed forward verification raise
`ArithmeticError`. Native integer status codes are not returned.

## Numerical methods

Ordinary tails use compiled incomplete-gamma kernels and preserve the smaller
probability. Quantiles evaluate only the inverse kernel for the smaller tail,
then divide the unit-rate coordinate by rate (or by x when solving rate).

When x*rate is subnormal or rounds to zero, the implementation keeps
log(z)=log(x)+log(rate) and evaluates the small-coordinate expansion
log(P)=shape*(log(z)-logGamma(1+shape)/shape). The omitted coordinate correction
is below floating-point precision in this regime. A Taylor expansion for
logGamma(1+shape)/shape avoids cancellation when shape is tiny; `expm1` preserves
small Q. Logarithmic inverse scaling also recovers representable answers whose
intermediate unit-rate quantile underflows.

For shape<1e-12 at normal or larger z, Q is evaluated as shape*E1(z), where
E1(z) is the exponential integral. Its relative approximation error is below
1e-9 over this branch. This avoids incomplete-gamma kernel defects at extremely
small shapes, including negative upper tails observed in SciPy 1.18.1.
Normal-coordinate inversions in this branch use 64 bisections in log(z) to solve
E1(z)=Q/shape; subnormal coordinates retain the logarithmic expansion.

Shape inversion uses the shared batched positive-parameter search, anchored at
z when z>=1 and otherwise at five, clipped to the search bounds. Starting near z
resolves large shapes whose CDF transition is too narrow for a generic geometric
search from five. Original batch indices remain aligned when endpoints resolve
before other rows. Every inverse is checked against the forward smaller tail,
with tolerance 1e-7 relative plus 32 minimum subnormal floats. Parameter accuracy
can be weaker when an input probability is subnormal and strongly quantized.

## Evidence

`tools/reference_dcdflib_gamma.py` compiles unchanged archived C and F77 sources.
The fixture records archive/source hashes, compiler commands, drivers and outputs;
no native implementation is distributed. Each language supplies 112 ordinary
cases (all status 0), six invalid-input cases and six wide-domain cases.

The wide cases expose native product underflow: x=rate=1e-200 and shape=0.5
returns P=0, although P is about 1.1283791670954814e-200. Another native inverse
reports status 10 for P=1e-200, shape=0.5, rate=1e-308, although x is about
7.85398163397433e-93. Both are recovered by logarithmic scaling.

The 302 tests include native comparisons, independent 120-digit integer-shape
finite sums, 400-digit tiny-shape series, half-shape error-function identities,
all inverse groups, endpoints, mixed batches, ownership and domain failures.
Tests include shapes down to 1e-320 and computed shapes up to 1e100. Tiny-shape
quantile checks verify the forward probability rather than assuming uniformly
accurate parameters after subnormal probability rounding.

[Batch measurements](dcdflib-gamma-benchmark.json) compare one array call with
repeated scalar calls to the same Python API, with answer agreement checked.
For 64/256 rows, measured speedups were about 39/106 for tails, 33/88 for x,
51/131 for shape, 34/92 for rate, and 13/16 for tiny-shape quantiles. These are
batching measurements, not comparisons with native C/Fortran execution.

A normal unit-rate coordinate can also yield a subnormal lower tail. If the
compiled kernel returns P=0 at 0<z<=1, a 32-term positive lower-gamma series
recovers P in logarithms. The omitted relative remainder is bounded by the
exponential-series tail beyond 32 terms (less than 1e-36 for z<=1). An independent
800-digit check of P(2,1e-160) verifies the representable tail near 5e-321.
