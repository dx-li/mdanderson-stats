# Legacy DCDFLIB F distribution

`cdff` and `cumf` implement the F-distribution contracts in the C and Fortran 77
DCDFLIB 1.1 libraries bundled with CDFLIB90. They include numerator and denominator
degrees-of-freedom inversions omitted from the F95 module. The existing `cdf_f`
family retains its F95 bounds and two-mode contract.

```python
from mdanderson_stats import cdff, cumf

p, q = cumf([0.1, 1, 10], dfn=5, dfd=10)
quantile = cdff(2, p=p, q=q, dfn=5, dfd=10).f
numerator_df = cdff(3, f=1, p=p[1], q=q[1], dfd=10).dfn
denominator_df = cdff(4, f=1, p=p[1], q=q[1], dfn=5).dfd
# Select both numerator-df roots for the same probability:
roots = cdff(3, f=5, dfd=5, p=0.94, df_bracket=([0.001, 0.43], [0.43, 100])).dfn
```

## Contract and Python conventions

| which | Computed group | Required inputs |
|---|---|---|
| 1 | p/q | f, dfn, dfd |
| 2 | f | p or q, dfn, dfd |
| 3 | dfn | p or q, f, dfd |
| 4 | dfd | p or q, f, dfn |

Omit the computed group. `DCDFLIBF` contains `which` and owned immutable broadcast
arrays `p`, `q`, `f`, `dfn`, `dfd`. The legacy paired-tail routine `cumf` returns
`(p, q)`. One probability may be omitted and reconstructed. When both are supplied,
the original three-machine-epsilon sum tolerance is enforced; the smaller tail
is preserved when normalizing the accepted pair.

Input f is finite and nonnegative, and input degrees of freedom are finite and
positive. **Input domains and search domains differ**: legacy validation has no
F95-style upper df/f bounds. Computed f is restricted to [0,1e100]; computed df
and df brackets retain [1e-100,1e100]. Forward inputs such as f=1e200 or df=1e200
are accepted when their numerical evaluation is representable. The tests cover
these wider inputs separately from the original search endpoints.

Python uses return values and exceptions in place of pointer/in-out mutation and
numeric status/bound outputs. Invalid inputs, absent/unidentified parameters and
unbracketed or out-of-range solutions raise `ValueError`. Failed kernels or
forward verification raise `ArithmeticError`. An unbracketed df request does not
prove no solution exists: the CDF can cross twice inside a same-sign bracket.
The native reference fixture retains actual status/bound diagnostics, including
status -1/-4/-5/-6/-3 for the tested invalid which/f/dfn/dfd/q inputs, and status
3 for an inconsistent probability pair. Native bound is defined only on failure;
the reference driver initializes it to zero for deterministic output otherwise.

Both probabilities must be positive for a df inversion at positive f. At f=0,
the lower probability is zero for all positive df and cannot identify either df.
A zero lower probability gives quantile zero; zero q has no finite quantile.
Rejecting unidentified/underflowed df targets gives a clear error instead of
returning an arbitrary parameter from a numerically flat CDF.

## Numerical method and root selection

The F CDF uses the beta coordinates x=dfn*f/(dfd+dfn*f), y=dfd/(dfd+dfn*f), with
shapes dfn/2 and dfd/2. Both coordinates are formed directly. If the product or
denominator overflows, or the product underflows before division, logarithms of
the input ratio supply a scaled coordinate pair without overflowing exponentials.
Zero f and equal-df f=1 have exact (0,1) and (.5,.5) tails. Unrepresentable positive
beta coordinates/shapes fail explicitly rather than being mistaken for exact
probability endpoints. The shared beta kernels preserve the smaller tail.

Quantiles invert both beta coordinates using the smaller input probability.
The ratio (dfd/dfn)*(x/y) has a log-domain fallback when its intermediate product
is invalid or underflows. Computed f can snap to its upper bound within eight
machine epsilons; input domains retain their stated semantics. An inverse must
reproduce the requested smaller probability within relative tolerance 1e-7 plus
32 smallest subnormals. Outside that tolerance it fails explicitly. These checks
are numerical consistency tests, not a guarantee of uniform accuracy for every
finite combination of extraordinarily large or small parameters.

Both df inversions accept `df_bracket=(lower, upper)`, whose endpoints broadcast
with the other inputs. The default is the source's [1e-100,1e100]. Endpoints must
be ordered, lie within that range, and straddle the desired probability or match
an endpoint. Smaller-tail matches within 32 machine epsilons preserve endpoint
roots. Numerically identical endpoint probabilities cannot identify a df.

Within a crossing interval, the search starts at five, clipped to the interval,
and expands locally by factors of five toward the crossing before performing
64 log-df bisections. At most 300 expansion steps are permitted. This avoids
jumping to extreme beta parameters for a nearby ordinary root, while retaining
the full search domain. Bisection preserves residual signs and retains the best
probability match; it does not assume df monotonicity. Floating reconstruction is
restricted to the established bracket, and the final answer is forward-verified.

For example, at f=5 and dfd=5, p=.94 selects two numerator-df roots; at f=.5 and
dfn=10, p=.15 selects two denominator-df roots. Tests select and verify both roots
and their surrounding sign changes. The full-domain endpoints have the same
residual sign in these cases. Like the native full-bound check, the default
request fails to bracket them; Python explicitly asks for `df_bracket`. Brackets
select one root and do not enumerate roots or locate tangencies without a sign
change. No uniqueness claim is made for a nonmonotone distribution parameter.

## Native and independent validation

`tools/reference_dcdflib_f.py` compiles the unchanged C implementation and machine
constants with its header, and all 64 unchanged Fortran source files in a separate
executable. It records the archive hash, every compiled source hash, compiler,
commands and drivers. The original sources and executables remain under ignored
research storage and are not distributed in the Python package. No source patch
is applied in either language.

Each language contributes **250 cases**: 64 tails and 62 each of f, dfn and dfd
inversion, plus six invalid-input cases. All forward and quantile cases return
status zero. The native df searches succeed in 47 numerator and 55 denominator
cases; 15 numerator and seven denominator requests return status 2 even though
the recorded generating parameters give roots. Python's explicitly selected
brackets recover those generating roots. Successful native inverse answers are
also checked by forward evaluation, allowing their arbitrary choice between
multiple roots. Both languages have the same status counts on this grid.

Independent validation uses the dfd=2 identity
p=(1+2/(dfn*f))**(-dfn/2), evaluated with log1p/expm1 for both tails, and reciprocal
symmetry with swapped degrees of freedom. These tests span df from 1e-100 to
1e100, quantile inversion, local searches with a fixed df as large as 1e100,
input df beyond the search bounds, scaled products, exact symmetry, small tails,
f=0/f=1e100, overlap with the F95 implementation, immutable ownership, empty
batches, strict probability-pair validation and invalid/unidentified requests.

[Batch timings](dcdflib-f-benchmark.json) compare one broadcast call with repeated
scalar calls to the same Python API and verify matching answers. On the recorded
machine, batches of 64/256 were about 23/61 times faster for tails, 36/51 times
for numerator-df inversion, and 29/47 times for denominator-df inversion. These
median-of-three measurements describe Python batching, not speedup over native
C or Fortran.

This resolves the two legacy F entry-point contracts. The [legacy noncentral-F port](dcdflib-nc-f.md) resolves its df inversions too.
The other legacy distribution contracts and public support routines
remain in the [CDFLIB90 completion checklist](cdflib90-coverage.md).
