# Legacy DCDFLIB chi-square

`cdfchi` and `cumchi` implement the archived C/F77 chi-square interfaces.
They are separate from the F95 `cdf_chisq` interface, whose df range is narrower.

```python
from mdanderson_stats import cdfchi, cumchi

r = cdfchi(x=3, df=2)
x = cdfchi(2, p=r.p, q=r.q, df=2).x
df = cdfchi(3, p=r.p, q=r.q, x=3).df
p, q = cumchi(3, 2)
```

## Contract and methods

Modes 1, 2 and 3 compute p/q, x and df. Omit the computed group. Input x must
be nonnegative finite and df positive finite; neither has an upper input bound.
Computed x is restricted to [0, 1e100] and computed df to [1e-100, 1e100], matching
the executable C/F77 searches. Degrees of freedom are real, not restricted to
integers. Results are owned, immutable broadcast arrays in `DCDFLIBChiSquare`.

Inversions accept either p or q, or both with a sum within three machine epsilons
of one. The smaller tail is preserved. All inversions require positive q.
Quantile p=0 returns zero; df inversion requires positive p and x because the
zero-coordinate endpoint does not identify df. Invalid input, unattainable
bounds and unrepresentable answers raise `ValueError`. Failed numerical kernels
or forward verification raise `ArithmeticError`, rather than returning a native
status integer. Quantile results within eight machine epsilons above the upper
bound are snapped to that bound.

The distribution is gamma with shape=df/2 and rate=1/2. Ordinary computations
reuse the [legacy gamma kernels](dcdflib-gamma.md), including their logarithmic
scaling, smaller-tail inversion and tiny-shape E1 approximation. The df search
uses the shared batched positive-parameter solver, starts at x when x>=1 (otherwise
five), and retains original row indices as endpoint solutions leave the search.
All inverse results are checked against the forward smaller tail with tolerance
1e-7 relative plus 32 minimum subnormal floats.

For df below twice the smallest normal float, computing df/2 first can introduce
large relative rounding error or zero. The implementation instead computes
Q = df * (E1(x/2)/2). Corrections are of relative order df times the logarithmic
coordinate, far below binary64 precision in this regime. When x/2 is subnormal,
E1 is evaluated as -EulerGamma-log(x)+log(2), avoiding coordinate rounding as
well. Inversion solves E1(z)=2*(q/df) using the shared 64-step logarithmic
bisection or the small-coordinate expansion, and restores x=2*z in logarithms.
The df=2 exponential identity uses `expm1`, with subnormal half-ulp ties
rounded downward because the exact CDF is strictly less than x/2. An independent
800-digit calculation verifies the correction, including x=three minimum floats. Quantized subnormal probabilities can identify only
a range of x values; tests check the recovered probability in that regime.

## Native and independent validation

`tools/reference_dcdflib_chisq.py` compiles the unchanged archived C and F77
sources. Each language contributes 85 ordinary cases: 30 paired tails, 30
quantiles and 25 df inversions, all with native status 0. Fixtures also retain
six invalid-input cases and eight wide-domain cases per language. Compiler
commands, source/archive hashes, drivers and unmodified-source declarations
are recorded. Native code and compiled binaries are not distributed.

The native wide cases show why status alone is insufficient:

- x=df=1e200 returns P=2, Q=0 and status 0 in both languages. Python returns
  P=Q=0.5, consistent with the large-df central limit.
- p=1e-100, df=1 returns x=0 with status 0. The representable answer is about
  1.5707963267948966e-200, independently checked by the df=1 normal identity.
- x=df=the smallest positive float returns P=0, Q=1 after both half-values
  round to zero. Python retains P=1 and Q about 1.84e-321, verified by a
  400-digit series with exact half-values.
- Inverting a median with df or x of 1e200 lies outside the source search bounds.
  Python rejects these cases even though forward evaluation accepts these inputs.

The 251 tests include native comparisons, independent 120-digit integer-shape
finite sums, error-function identities, a 400-digit lower-gamma series, exact
exponential behavior, all inversions, large-df limits, mixed endpoint batches,
empty inputs, owned arrays and invalid/unrepresentable cases.

[Batch measurements](dcdflib-chisq-benchmark.json) compare array calls with
repeated scalar calls to the same Python API and verify answer agreement.
They measure Python batching, not speed relative to native C/Fortran routines.

For 64/256 rows, the recorded speedups are about 33/96 for tails, 30/58 for
x inversion, 47/107 for df inversion and 15/19 for tiny-df quantiles.
