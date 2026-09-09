# CDFLIB incomplete-gamma support

`gratio(a,x,ind=0)` and `grat1(a,x,r,eps=5e-15)` return tuples of independently
owned, immutable float64 arrays `(p,q)`. Every input broadcasts.

```python
from mdanderson_stats import grat1, gratio, rcomp

p, q = gratio(20, 20)
tiny_p, tiny_q = gratio(1e-100, 1e-300)
subnormal_p, _ = gratio(200, 2)
_, subnormal_q = gratio(1, 744)
small_p, small_q = grat1(0.1, 2, rcomp(0.1, 2))
```

## Contracts

`gratio` accepts finite nonnegative a,x except (0,0). At a=0,x>0 it returns
(1,0); at a>0,x=0 it returns (0,1). `ind` accepts signed int32 values, including
broadcast arrays. All selectors now use full float64 accuracy, satisfying the
source's less-demanding modes without reproducing their sign errors.

`grat1` requires 0≤a≤1, x≥0, a finite nonnegative r, and finite positive eps.
The caller normally supplies r=exp(−x)x^a/Gamma(a), as returned by `rcomp`.
As in F95, the continued-fraction branch consumes r when a>0, a≠1/2 and x≥1.1.
Other branches ignore r. This preserves the source's observable input contract:
scaling r scales q only in that branch. Unlike `gratio`, `grat1(0,0,r)` preserves
the native (0,1) result. An inconsistent r that produces probabilities outside
[0,1] raises `ArithmeticError`.

Invalid domains raise `ValueError`. The continued fraction uses at least the
accuracy associated with tolerance 1e-14; requested tolerances below four machine
epsilons use that practical float64 floor. No interface promises accuracy beyond
float64 resolution. Iteration is bounded, and failed numerical checks raise
`ArithmeticError` instead of returning uninitialized outputs or NaNs.

## Numerical implementation

Zero endpoints are detected from the individual inputs, avoiding the source's
underflowing a*x test. For 0<x≤1.1, the defining lower-integral series is written
as x^a/Gamma(1+a) times (1+aH). The alternating series for H takes 40 terms;
factoring a out of the complete log probability preserves subnormal shapes.
Stable log-gamma remainders and log1p/expm1 retain tiny complementary tails.

For x>1.1 and a<1e-16, Q≈a E1(x) reuses the exponential-integral strategy already
used by the legacy gamma implementation, with a tighter small-shape threshold.
The ordinary region reuses the existing compiled incomplete-gamma wrapper.
It computes the smaller tail directly and obtains the larger one by complement.

Subnormal tails receive explicit recovery. Upper tails use `rcomp` multiplied by
a rescaled continued fraction, with a 1,000-iteration limit. Lower tails use the
endpoint expansion of the defining integral

```text
P/r = integral_0^infinity exp(-a*t + x*(1-exp(-t))) dt.
```

Scaling t by 1/(a−x) avoids a prohibitively long positive series near a large
shape. The implementation integrates the first 32 expansion terms analytically,
checks the expansion region and the final three terms, and multiplies the gamma
factor only at the end. This retains lower tails that the compiled kernel rounds
to zero. Existing distribution interfaces and their implementations are unchanged.

## Evidence and limits

[The native audit](cdflib-incomplete-gamma-reference.md) records the source's
large-shape bias, center-branch sign errors, product-underflow endpoints and
invalid-tolerance behavior. [Port tests](../tests/test_cdflib_incomplete_gamma.py)
cover every applicable native case against independent values, selector and r
semantics, branch transitions, extreme inputs, subnormal tails, broadcasting and
immutability. Independent checks use high-precision gamma integrals, integer
finite sums, and separately integrated scaled lower tails at shapes 1e20 and
1e30. Numerical accuracy is tested, not inferred from agreement with defective
native outputs.

[Benchmarks](cdflib-incomplete-gamma-benchmark.json) compare batches of 64 and 256
coordinates with repeated scalar calls, checking identical results. They include
ordinary, tiny-shape, large-center, lower/upper-subnormal and `grat1` fraction
cases. These measure batching benefits, not speed relative to Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
