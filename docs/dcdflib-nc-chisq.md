# Legacy DCDFLIB noncentral chi-square

`cdfchn` and `cumchn` implement the archived C/F77 noncentral chi-square
interfaces, supplementing the separately bounded F95 `cdf_nc_chisq` API.

```python
from mdanderson_stats import cdfchn, cumchn

r = cdfchn(x=0.2, df=10, pnonc=20)
x = cdfchn(2, p=r.p, df=10, pnonc=20).x
df = cdfchn(3, p=r.p, x=0.2, pnonc=20).df
nc = cdfchn(4, p=r.p, x=0.2, df=10).pnonc
p, q = cumchn(0.2, 10, 20)
```

## Contract

Modes 1–4 compute p/q, x, df and pnonc respectively. Omit the computed group.
Input x and noncentrality are nonnegative finite, and df is positive finite,
without upper input caps. Computed x lies in [0,1e100], df in [1e-100,1e100],
and noncentrality in [0,1e4]. Noncentrality is the sum of squared normal means.
Results are owned, immutable broadcast arrays in `DCDFLIBNoncentralChiSquare`.

During inversion, p is required in [0,1-1e-16], including the upper endpoint
accepted by the executable. Input q is ignored completely, including its value,
type and shape. Returned inverse q is 1-p, following the existing Python legacy
noncentral-F convention instead of retaining a meaningless native placeholder.
Forward evaluation computes both tails. The p=0 quantile is exactly zero;
parameter inversion requires positive x and p, rejecting unidentified endpoint
parameters and finite answers produced only by native underflow.

Invalid input, unidentified parameters, unattainable search bounds and
unrepresentable quantiles raise `ValueError`. Numerical-kernel failures and
inverse answers that fail forward verification raise `ArithmeticError`.
The API does not promise that every finite extreme parameter combination or
inverse target is resolvable in double precision. General enormous noncentrality
near the distribution's center can still exceed the numerical kernels, even
though tail bounds and the one-degree identity resolve useful wide cases.

## Numerical methods

Zero noncentrality uses the independently validated [legacy central chi-square
kernel](dcdflib-chisq.md), preserving its wide-shape and subnormal repairs.
Positive noncentrality is not rounded to zero at the native 1e-10 threshold.
Ordinary cases use direct lower and upper noncentral chi-square kernels.
The installed SciPy 1.18.1 source was inspected after documentation lookup:
`ncx2` accepts all positive finite real df, its CDF/quantile use `chndtr` and
`chndtrix`, and the upper tail uses the compiled complementary kernel. The
integer-df normal-sum interpretation in the tutorial does not restrict df to
integers in the actual implementation.

Before general evaluation, Chernoff bounds prove some tails unrepresentably
small. At t=1/2 in the Laplace transform and t=1/4 in the moment generating
function, respectively,

```text
log P(X<=x) <= x/2 - (df/2)*log(2) - pnonc/4
log P(X>=x) <= -x/4 + (df/2)*log(2) + pnonc/2
```

A tail is replaced by zero only when its bound is below half the minimum
positive float. This resolves the audited large-noncentrality timeout example
without a normal approximation or a fabricated probability.

For positive x<1e-80 after this elimination, the leading Poisson term gives
P=exp(-pnonc/2)*P_central. Higher terms have relative corrections controlled by
pnonc*x/(2*(df+2)), far below floating-point precision in the surviving region.
Q is computed as -expm1(-pnonc/2)+exp(-pnonc/2)*Q_central, retaining tiny central
and noncentral contributions. This also preserves the tiny-df limiting mass
at every representable positive x while keeping P(0)=0 exactly.

For positive noncentrality <=1e-12 outside that small-x region, four positive
Poisson terms retain both central and noncentral tail contributions. After the
Chernoff elimination, the omitted terms are negligible at double precision;
400-digit exponential-integral checks cover the subnormal upper-tail case
where the compiled kernel loses the central contribution.

For df=1, X=(Z+sqrt(pnonc))**2. Both tails use paired normal probabilities at
sqrt(x)-sqrt(pnonc) and -sqrt(x)-sqrt(pnonc). The difference of roots is evaluated
as (x-pnonc)/(sqrt(x)+sqrt(pnonc)). When those normal integration endpoints are
nearby, a local density integral with the correction (pnonc-1)*x/6 avoids
cancellation; its region sqrt(x)*(sqrt(pnonc)+1)<1e-4 bounds the next correction
at roughly floating-point precision. The smaller resulting tail defines the
final complementary pair.

Quantiles use a compiled initial estimate followed by bounded batched search
and verification; invalid initial estimates use a mean-based starting value.
Df inversion searches its legacy positive bounds. Noncentrality inversion
searches 1+pnonc over [1,10001], including the zero endpoint. Resolving increments
below floating-point spacing near one is not guaranteed by this coordinate.
Mean-based starting values preserve large central roots, and original row
indices remain aligned when endpoint rows resolve early.

Every inverse is checked against the smaller tail with relative tolerance 1e-7,
plus half the spacing of input p and 32 minimum subnormal floats. The spacing
term accounts for information lost by the legacy p-only contract near one; it
does not treat the ignored q as additional information. Input-probability and
parameter quantization can limit inverse accuracy.

## Validation and performance

The unchanged C/F77 [reference audit](dcdflib-nc-chisq-reference.md) retains 144
ordinary calls per language plus invalid, ignored-q, boundary and wide cases.
The source hashes, compiler commands, drivers and explicit timeout records are
preserved. No archived numerical source or executable is bundled.

The 388 implementation tests include all ordinary references, 120-digit Poisson
mixtures, 800-digit leading-term checks, shifted-normal identities, tiny positive
df down to the minimum float, empty/mixed batches, ownership and search bounds.
Tests explicitly preserve independent accuracy where native roundtrips are
wrong: at x=0.2,df=10,pnonc=20, the lower probability is about 4.10301409e-12,
not the native 1.316e-26. At x=1e-100,df=2,pnonc=4, the leading tail near
6.76676e-102 survives; the corresponding subnormal case is also retained.

[Batch measurements](dcdflib-nc-chisq-benchmark.json) compare one array call with
repeated scalar calls to this Python API and verify answer agreement. They
measure Python batching, not speed relative to native C or Fortran.
For 64/256 rows, recorded speedups were about 39/126 for tails, 53/139 for
quantiles, 49/125 for df and 54/142 for noncentrality inversion.

See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
