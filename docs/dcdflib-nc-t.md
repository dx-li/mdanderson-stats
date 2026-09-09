# Legacy noncentral t

`cdftnc` and `cumtnc` implement the signed C/F77 noncentral-t distribution
contracts in the pinned CDFLIB90 archive. For independent Z~N(0,1) and
V~chi-square(df), T=(Z+pnonc)/sqrt(V/df). Noncentrality is a signed normal
mean. It is not the squared-shift parameter used by noncentral chi-square/F.

```python
from mdanderson_stats import cdftnc, cumtnc

p, q = cumtnc([-2, 0, 2], 2, -3)
quantiles = cdftnc(2, p=[0.1, 0.5, 0.9], df=2, pnonc=-3).t
roots = cdftnc(3, p=0.03, t=1, pnonc=3, df_bracket=([0.001, 0.2], [0.2, 100])).df
shift = cdftnc(4, p=0.5, t=1, df=2).pnonc
```

## Contracts

| Mode | Omit/computed group | Computed domain |
|---|---|---|
| 1 | p and q | Complementary probabilities |
| 2 | t | [-1e100,1e100] |
| 3 | df | [1e-100,1e4] |
| 4 | pnonc | [-1e4,1e4] |

Input t and pnonc are arbitrary finite signed numbers; input df is positive
finite. Only computed values have the bounds above. The df upper bound is
**1e4 in both executables**, despite a header claiming 1e10 and a failure status
reporting bound=1e100. The [native reference audit](dcdflib-nc-t-reference.md)
records these discrepancies.

Inverse p is required in [0,1-1e-16]. The native inversion ignores q entirely,
so Python also ignores inverse q, including its type and shape. Returned inverse
q is 1-p; the source instead leaves the ignored placeholder unchanged. p=0 is
recognized but rejected because it identifies no finite exact quantile or
parameter solution. Mode 1 computes both tails and requires both output inputs
omitted. Mode selection, required inputs and finite domains are validated.

All numerical inputs broadcast. Results own immutable arrays, including copied
input parameters; empty batches are supported. `cumtnc` returns the paired tails.
Invalid or unidentified inversions raise `ValueError`; unresolved numerical
failures raise `ArithmeticError` rather than returning a native status-zero
false success.

Df can have multiple roots. `df_bracket` selects a sign-changing interval within
the executable bounds. The default is the whole domain. Neither choice
enumerates all roots or detects tangencies without a crossing. At t=0 df is
unidentified and rejected. A local bracket can recover the two roots missed by
the native global search at p=0.03,t=1,pnonc=3.

## Numerical implementation

Ordinary noncentral tails use the compiled SciPy survival kernel and signed
reflection, computing both tails directly and retaining the smaller one.
The installed SciPy 1.18.1 source was inspected: `nct.cdf` and `nct.sf` currently
use different implementation paths despite the general class documentation.
The package does not depend on that internal detail as a public API.
Central inputs use the independently validated wider legacy Student-t kernel.
At t=0 the exact identity F(0)=Phi(-pnonc) applies. No threshold rounds a small
nonzero noncentrality to zero.

The existing conditional-normal repair handles ordinary negative t with a small
lower tail. Additional logarithmic integration handles tiny df, large coordinates,
large noncentrality and nonfinite compiled results. With a positive normal
numerator z, the conditional chi-square coordinate is df*(z/t)^2. It is formed
in logarithms so neither squaring nor division need be representable. The
negative-t lower tail integrates the lower chi-square tail against phi(z+nc).
For positive t, the lower probability is Phi(-nc) plus the upper chi-square
integral against phi(z-nc); the upper probability integrates the lower tail.

Integration factors out a sampled log scale and checks quadrature convergence
and its error estimate. Positive-numerator integration uses a normal coordinate
within 40 standard deviations; omitted normal mass is below the float range.
For extremely small chi-square coordinates, the leading lower-gamma expression
has a relative correction below float precision. For tiny df, the upper-gamma
E1 expansion retains contributions even when df/2 itself rounds to zero.
For df<1e-50 and |nc|<=8, the finite-coordinate correction to Phi(-nc) is
negligible relative to the smaller tail. This shortcut is deliberately limited:
for nc=40, a positive-coordinate gamma contribution can dominate Phi(-40) and
must remain, including subnormal probabilities.

At df>=1e20, a conditional-normal expansion retains denominator fluctuations
through scale=hypot(1,t/sqrt(2*df)) and mean correction -t/(4*df). It also
includes the first Edgeworth correction, whose standardized third cumulant is
t**3/(4*df**2*scale**3). Simply using Phi(t-nc), or dropping this skewness term,
would lose accuracy when both t and nc are enormous. After this correction,
the omitted standardized terms are of order 1/df times polynomials in the normal
coordinate, below the 1e-7 inverse tolerance over representable normal tails at
this threshold. This is an asymptotic computation, not an exact finite-df identity.
It also avoids rounding a huge conditional chi-square coordinate by more than
its effective probability resolution.

Quantiles and signed noncentrality inversions first resolve zero and exact
central/normal cases, then search a positive magnitude on the appropriate side
of zero. Df uses a sign-preserving positive bracket search. Initial compiled
quantile estimates are only starting points: final results must pass an
independent forward evaluation. Checks retain the smaller tail and allow
relative error 1e-7, half a spacing of the supplied p, and 32 minimum subnormal
units. Probability rounding or flat regions can make parameter coordinates
less accurate than the forward probability; numerically indistinguishable
zero solutions use zero. This does not assert a unique resolved parameter in
an arbitrarily ill-conditioned problem.

## Validation and performance

The implementation tests exercise every ordinary call from both unchanged native
language fixtures, with local df brackets and forward checking of inverse
results. Independent checks include the df=2 normal-CDF closed form, zero-coordinate
normal probabilities, Chebyshev conditional-normal bounds at enormous df,
independent normal/chi-square Simpson integration with grid refinement,
truncated normal second moments for extreme tails and quantiles, and 800-digit
Decimal tiny-df expansions preserving minimum-subnormal contributions.
Signed reflection, bounds, ignored q, immutability, broadcasting and empty arrays
are covered. Native zero-quantile errors are repaired rather than reproduced.

`tools/benchmark_dcdflib_nc_t.py` records batch/scalar equivalence and timings
in [the benchmark artifact](dcdflib-nc-t-benchmark.json). It compares a broadcast
call with repeated calls to this same Python API, not with compiled native
executables. Ordinary calculations and searches batch in NumPy/SciPy; difficult
conditional integrations remain scalar per affected row. Exceptional sharp or
ill-conditioned integrals can still raise an explicit numerical error.

All twelve legacy distribution families now have separate C/F77 evidence and
Python interfaces. CDFLIB90 remains partial because its public support, numerical
helper, root-finder/state and console/string/sort interfaces are still under review.
