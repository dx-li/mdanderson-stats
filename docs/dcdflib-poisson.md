# Legacy DCDFLIB Poisson

`cdfpoi` and `cumpoi` independently implement the archived C/F77 Poisson
interfaces, including zero mean and input domains wider than F95 `cdf_poisson`.

```python
from mdanderson_stats import cdfpoi, cumpoi

r = cdfpoi(s=2.5, mean=3)
s = cdfpoi(2, p=r.p, q=r.q, mean=3).s
mean = cdfpoi(3, p=r.p, q=r.q, s=2.5).mean
p, q = cumpoi(0, 0)  # 1, 0
```

## Contract

Modes 1, 2 and 3 compute p/q, count s and mean. Omit the computed group.
The source XLAM parameter is named `mean`. Inputs s and mean are nonnegative
finite values without an upper cap. Computed s and mean are bounded by
[0,1e100], matching the executable native searches. `DCDFLIBPoisson` contains
owned, immutable broadcast arrays.

Counts are real-valued: the implementation uses the source's continuous gamma
extension. Count inversion solves that continuous equation and does not round
to an integer Poisson quantile. At mean=0, every valid count has p=1, q=0.

Inversions accept either p or q, or a supplied pair summing to one within three
machine epsilons; the smaller tail is retained. Native inversion rejects q=0,
so the zero-mean endpoint is available for forward evaluation only. Python also
requires p>0: a finite mean with nonnegative count has strictly positive lower
probability, and returning a point where that probability numerically underflows
does not identify a mathematical solution. Count inversion requires positive
mean and rejects probabilities below the nonnegative-count range. It does not
clamp an unattainable negative count to zero.

Invalid input, unattainable bounds and unrepresentable results raise `ValueError`.
Numerical-kernel failures or failed forward verification raise `ArithmeticError`.
The interface returns results rather than native integer status codes.

## Methods

P is the upper gamma ratio Q(s+1,mean); the Poisson upper tail is the lower gamma
ratio. Direct reduction avoids the native intermediate doubling of s+1 and mean,
which overflows for some valid finite inputs. Zero-count tails use exp(-mean)
and -expm1(-mean) directly. The [shared legacy gamma kernels](dcdflib-gamma.md)
retain small complementary tails, logarithmic scaling and selective recovery of
subnormal lower-gamma probabilities lost by the compiled kernel.

Mean inversion uses the inverse gamma kernel for the smaller probability.
Results within eight machine epsilons above 1e100 are snapped to that bound.
Count inversion searches shape=s+1 over [1,1e100], then subtracts one. Its batched
search starts at mean when mean>=1, otherwise five, and preserves original row
indices as endpoint solutions leave the search. The shape representation limits
resolution of counts very close to zero, as in the source's s+1 reduction.
Every inverse is checked against the forward smaller tail with tolerance 1e-7
relative plus 32 minimum subnormal floats. Subnormal probability quantization
can limit inferred parameter accuracy; tests distinguish this from tail accuracy.

## Evidence

`tools/reference_dcdflib_poisson.py` compiles unchanged archived C and F77 sources.
Each language supplies 78 ordinary cases: 30 paired tails, 24 count inversions
and 24 mean inversions, all reporting status 0. Six invalid-input and eight
wide-domain cases are also retained. Fixtures record compiler commands, drivers,
source/archive hashes and the absence of source adaptations. No native source
or compiled binary is distributed.

Both native languages return NaN tails at s=mean=1e308, and P=0, Q=2 at
s=mean=1e200, despite status 0. Direct gamma reduction gives P=Q=0.5 at both
scales, consistent with the large-mean limit. The native mean inverse at
s=0, q=1e-100 returns mean=0 with status 0; the repaired result is about 1e-100.
At p=0, q=1, s=0, native inversion reports success with an arbitrary finite mean
above 1000. Python rejects this underflow-based answer.

The 219 Poisson tests include native comparisons, independent 120-digit finite
Poisson sums, half-integer error-function recurrences, 800-digit small-mean sums,
all inversions, zero mean, large-mean limits, mixed endpoint batches, ownership,
empty arrays and invalid domains. The small-mean tests exposed a pre-existing
shared gamma-kernel underflow at shape=2, coordinate=1e-160. A separate regression
and selective positive series repair now recover its probability near 5e-321.

[Batch measurements](dcdflib-poisson-benchmark.json) compare one array call with
repeated scalar calls to the same Python API, with answer agreement checked.
They measure Python batching, not performance relative to native C/Fortran.

For 64/256 rows, recorded speedups were about 35/121 for paired tails,
64/157 for count inversion and 35/89 for mean inversion.
