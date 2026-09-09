# Legacy DCDFLIB noncentral F

`cdffnc` and `cumfnc` implement the archived DCDFLIB 1.1 C/F77 noncentral-F
contracts, separately from the [F95 interface](cdflib90.md). Noncentrality is
the sum of squared normal shifts in the numerator chi-square. It is not a
standard deviation or a shift of the F statistic.

```python
from mdanderson_stats import cdffnc, cumfnc

p, q = cumfnc(1, 2, 10, 4)
quantile = cdffnc(2, p=p, dfn=2, dfd=10, pnonc=4).f
numerator_df = cdffnc(3, p=p, f=1, dfd=10, pnonc=4, df_bracket=(1.9, 2.1)).dfn
denominator_df = cdffnc(4, p=p, f=1, dfn=2, pnonc=4, df_bracket=(9, 11)).dfd
noncentrality = cdffnc(5, p=p, f=1, dfn=2, dfd=10).pnonc
```

## Contract

| `which` | Computed group | Required input |
|---|---|---|
| 1 | p and q | f, dfn, dfd, pnonc |
| 2 | f | p, dfn, dfd, pnonc |
| 3 | dfn | p, f, dfd, pnonc |
| 4 | dfd | p, f, dfn, pnonc |
| 5 | pnonc | p, f, dfn, dfd |

Omit the computed group. `DCDFLIBNoncentralF` contains `which` and owned,
immutable broadcast arrays `p`, `q`, `f`, `dfn`, `dfd`, `pnonc`. Empty batches
are supported. `cumfnc` returns a pair of lower/upper tail arrays.

Unlike `cdff` and the F95 probability-pair interfaces, **legacy `cdffnc` ignores
input q during inversion**. Python requires p and ignores q's value and shape,
including nonfinite or inconsistent values. It cannot infer p from q. The
native routine leaves the unused q placeholder untouched; Python deliberately
returns q=1-p for inversions so the result contains a meaningful complement.
Forward evaluation computes both tails directly and preserves the smaller one.

The executable legacy p check permits 0 through `1-1e-16`, inclusive; the header
incorrectly prints a half-open interval. p=1 is invalid for inversion. p=0 gives
f=0 in quantile mode. Parameter inversions require positive f and p, rejecting
zero or numerically unidentified cases rather than manufacturing an arbitrary
parameter.

Finite input f and pnonc are nonnegative, and input df are positive, without
upper input limits. Computed f is bounded by [0,1e100], computed df by
[1e-100,1e100], and computed pnonc by [0,1e4]. Numerical kernels can still fail
within mathematically valid input domains; such failures raise `ArithmeticError`.
Invalid inputs, unbracketed/unidentified roots and out-of-bounds solutions raise
`ValueError`, replacing native status/bound output arguments.

The F95 `cdf_nc_f` remains distinct: which=3 computes noncentrality, and its
input domains remain f<=1e100, df in [1e-3,1e10], pnonc<=1e4. Its paired-tail
contract can preserve smaller upper probabilities than legacy p-only inversion.

## Numerical implementation

Positive-noncentrality tails generally use public SciPy ncf CDF/SF kernels.
Exact zero noncentrality uses the wide-domain central F kernel, avoiding the
SciPy zero-noncentrality survival-function defect. For dfd=2 the Poisson-beta
mixture sums exactly to

```text
log(p) = -(dfn/2)*log1p(2/(dfn*f)) - pnonc/(dfn*f+2).
```

The shared kernel evaluates this identity using scaled/logarithmic terms and
`expm1`, preserving small tails even when intermediate products overflow or
underflow. This also repairs failed quantile refinement for very large dfn.
The F95 interface shares this numerical improvement without changing its domains.

Quantiles use the smaller-tail inverse kernel and bounded refinement if its
answer fails forward verification. Noncentrality uses a vectorized bounded
search. Both df inversions share the independently written legacy F search:
endpoint checks, a factor-five local search starting at five, then 64 logarithmic
bisections retaining the best probability residual. Fixed parameters remain
aligned through original row indices when endpoint solutions leave the search.
Every inverse is checked against the smaller probability tail with relative
tolerance 1e-7, an allowance of half one spacing of input p for its unavoidable
rounding, and 32 minimum subnormals. The same p-rounding allowance applies to
endpoint comparison. The F95 interface retains its stricter paired-tail check.

F probabilities need not be monotone in either df. Optional broadcast
`df_bracket=(low,high)` is restricted to df modes and ordered endpoints inside
[1e-100,1e100]. A bracket selects a sign-changing root; it does not enumerate roots,
find tangencies, or establish uniqueness. Same-sign full-range endpoints can
reject a request even when two interior roots exist. Tests select both roots for:

- dfn: f=5, dfd=5, pnonc=0.5, p=0.77, brackets [0.001,0.045] and [0.045,100].
- dfd: f=0.5, dfn=10, pnonc=4, p=0.1, brackets [0.001,0.65] and [0.65,100].

## Native and independent evidence

`tools/reference_dcdflib_nc_f.py` compiles unchanged archived C sources/header
and all 64 unchanged F77 sources in separate executables. The fixture records
archive/source hashes, compilers, commands, drivers and empty adaptation lists.
Neither archived source nor native executables are distributed with the package.

Each language contributes 378 ordinary cases: 81 tails, 81 f inversions,
81 numerator-df inversions, 81 denominator-df inversions and 54 noncentrality
inversions. Each also has seven invalid-input and four ignored-q cases. Both
languages return the same status counts: all tails, quantiles and noncentrality
cases report success; df searches report success for 49 numerator and 43
denominator cases, with respectively 32 and 38 status-2 failures despite known
generating roots. Explicit Python brackets recover those roots.

Native summation differs from accurate tails by up to about 1.82e-5 on this grid.
Inverting those approximate input probabilities can shift noncentrality by about
0.36%. Fixtures retain these differences; tests verify Python answers against the
actual input probabilities rather than copying displaced native parameters.

There is also a **false native success** for numerator-df inversion at f=0.1,
dfd=0.5, pnonc=4, p=0.009057089161749006, generated using dfn=10. Both native
implementations return dfn approximately 2.431305206676111e-9 with status zero.
The CDF there is approximately 0.13533527914095073. An independent positive
lower bound using only the first term of the j=0 incomplete-beta mixture exceeds
0.13, disproving that native answer without relying on the Python CDF kernel.

Independent 120-digit Decimal Poisson-beta sums check tails and all four inverse
modes. Additional checks cover dfd=2 closed forms over df 1e-100..1e100, wider
central inputs, input noncentrality above 1e4, exact noncentrality endpoints,
multiple roots, mixed endpoint/interior batches, p rounding near one, ignored-q
shape, immutable ownership, empty inputs, invalid requests and F95 overlap.

[Batch measurements](dcdflib-nc-f-benchmark.json) compare vectorized calls against
repeated calls to the same Python API, with answer agreement checked. They are
not comparisons against native C or Fortran. On the recorded machine, batches
of 64/256 were about 39/103 times faster for tails, 45/97 times for numerator-df
inversion, and 42/84 times for denominator-df inversion (median of three runs).

This resolves the two legacy noncentral-F distribution entry points.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
