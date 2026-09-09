# CDFLIB gamma scaling factor

`rcomp(a,x)` computes exp(−x)x^a/Gamma(a), the scaling factor used by the
incomplete-gamma support routines. Finite inputs broadcast through NumPy; results
are independently owned, immutable float64 arrays.

```python
from mdanderson_stats import rcomp

ordinary = rcomp([0.5, 2, 20], [1, 2, 20])
large_center = rcomp(1e308, 1e308)
tiny_shape = rcomp(1e-309, 0.1)
negative_shape = rcomp(-0.5, 1)
```

The F95 return of zero for x≤0 is preserved. For x>0, the implementation uses
the real reciprocal-gamma continuation: nonpositive integer shapes give zero,
and negative nonintegers retain the gamma sign. This makes the mathematical
definition explicit beyond the source's negative-shape example, where its
internal `gam1` approximation has a restricted domain. Nonfinite inputs raise
`ValueError`; a true output overflow raises `ArithmeticError`. Underflow keeps
the sign of the result.

## Numerical implementation

Near a=0, the recurrence Gamma(1+a)=a Gamma(a) permits multiplication by a
after exponentiation. The existing local log-gamma approximation preserves the
small increment, including negative and subnormal shapes.

For a≥8, a Stirling representation gives the log factor as

```text
0.5 log(a) − 0.5 log(2π) − delta(a) − [x−a−a log(x/a)].
```

The existing ten-term Stirling correction evaluates delta. Close to x=a, the
last bracket uses a stable log1p remainder with d=(x−a)/a. Subtracting the
original coordinates avoids losing a meaningful deviation by first rounding
x/a. Away from the center, separate logarithms avoid underflow of the quotient.
Combining the complete log factor before exponentiation retains small results
that an underflowed intermediate exponential would discard.

Other shapes use the existing positive log-gamma helper or SciPy's documented
real log-absolute-gamma and sign functions. No dependency or distribution kernel
changed. As with float64 log-domain calculations generally, cancellation can
reduce relative accuracy for extreme negative noninteger shapes; the interface
does not promise correct rounding at every real input.

## Validation and performance

[Tests](../tests/test_cdflib_gamma_factor.py) compare every native `rcomp` audit
case against independent values and cover huge centers, close coordinates,
subnormal results, negative-shape continuation, reciprocal-gamma zeros, overflow,
broadcasting and immutability. The independent calculation uses 800-digit
Decimal arithmetic, gamma recurrences and half-integer reflection. The
[native audit](cdflib-incomplete-gamma-reference.md) retains source provenance.

[Benchmarks](cdflib-gamma-factor-benchmark.json) compare batches of 64 and 256
coordinates against repeated scalar calls, verifying identical results. Cases
include ordinary, tiny, large and negative shapes. The measured speedups describe
batching benefits, not a comparison against Fortran.

The later [paired beta integral](cdflib-bratio.md) completes all 35 F95
mathematical procedures. Their [constants](cdflib-constants.md) are also implemented.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
