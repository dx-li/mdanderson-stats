# Legacy noncentral chi-square reference audit

This audit establishes unchanged C/F77 evidence for the implemented [`cdfchn` and
`cumchn` interfaces](dcdflib-nc-chisq.md). The existing `cdf_nc_chisq` implements
the F95 contract;
it does not establish the legacy input and inversion contracts by itself.

## Contract differences

Modes 1–4 compute p/q, x, df and pnonc respectively. Noncentrality is the sum
of squared normal means. The source's second which=3 comment is a typo: the
executable uses which=4 for noncentrality inversion.

Inputs x and pnonc are nonnegative finite without upper caps; df is positive
finite without an upper cap. Computed x spans [0,1e100], df [1e-100,1e100],
and pnonc [0,1e4]. The executable accepts p in [0,1-1e-16], including its
upper endpoint despite the header's half-open interval. During inversions,
q is ignored: it is neither validated nor used, and its input value survives
unchanged. This differs from the F95 Python paired-tail inversion contract.
The new legacy API must explicitly document its returned-q convention.

For x<=0, the tail routine returns P=0,Q=1. For pnonc<=1e-10, it calls the
central chi-square kernel. Otherwise it sums a Poisson mixture of central
chi-square tails, starting near the largest Poisson weight. It stops a direction
when its new term is below 1e-5 times the running sum **or when that sum is below
1e-20**. The latter condition can stop before the dominant probability terms
are reached. The upper tail is obtained by subtracting the lower tail from one.
All inverse searches start at five, using absolute tolerance 1e-50 and relative
tolerance 1e-8 against P alone.

## Reproduction and validation

`tools/reference_dcdflib_nc_chisq.py` compiles the unchanged two C sources and
header, and all 64 F77 sources. Fixtures preserve archive/source hashes,
compiler versions and commands, driver text and empty source-adaptation lists.
Drivers initialize status to -999 and bound to zero; bound is undefined by the
native contract on success. No archived source or executable is bundled.

Each language records 144 ordinary calls: 36 tails and 36 calls for each inverse.
The grid uses x=0.2,2,10,30; df=0.5,2,10; and noncentrality=0.5,4,20.
All ordinary calls return status 0. There are also seven invalid cases, four
ignored-q cases, eight boundaries and nine wide cases per language. Wide and
special calls have a three-second timeout, recorded separately from statuses.

The 342 tests evaluate ordinary calls against the existing Python API, check
all inverse Python answers against the actual requested P, and use independent
120-digit Poisson mixtures for the 48 integer-shape native references. The
ordinary native absolute CDF error is below 2e-6, but that bound must not be
mistaken for relative small-tail accuracy or parameter accuracy. Separate tests
explicitly refute native false successes with positive-term lower bounds.

## False successes and endpoint ambiguities

At x=0.2,df=10,pnonc=20, both native sources return P approximately
1.3160092139386282e-26. The independent Poisson mixture gives approximately
4.10301409e-12, more than fourteen orders larger. Native inverses reproduce the
original parameters because they invert the same inaccurate series. For the
actual requested P, the Python noncentrality inverse is approximately 87.8100,
not the native value 20. This example demonstrates why native roundtrips alone
cannot validate the statistical calculation.

At x=1e-100,df=2,pnonc=4, native P is approximately 3.38338e-202.
The zero-th Poisson term alone gives the rigorous lower bound
exp(-2)*(1-exp(-x/2)), approximately 6.76676e-102. An 800-digit calculation
also proves a positive representable bound at x=1e-320, where native P is zero.
These cases need small-tail evaluation that does not drop the leading term.

For P=1e-100,df=2,pnonc=0, native x inversion returns zero with status 0;
the central exponential identity gives x=-2*log(1-P), approximately 2e-100.
The existing legacy central chi-square implementation returns that value.
For the same probability with pnonc=4, native x is approximately 5.23942e-50.
Its zero-th Poisson term alone exceeds the target by over forty orders of
magnitude, independently refuting the returned success.

At x=0,df or pnonc inversion with P=0 returns the arbitrary initial value five.
At positive x with P=0, native underflow produces finite parameters near 395
with status 0 even though no finite positive df and finite noncentrality yield
an exact zero tail there. The existing F95 Python API rejects these inversions.
The p=0 quantile correctly returns x=0.

## Wider input requirements

With central x=df=1e200 or 1e308, both languages return the invalid probability
pair P=2,Q=0 with status 0. The central reduction must preserve the already
validated legacy chi-square repairs. Both languages exceed the three-second
timeout at x=1,df=2,pnonc=1e20; the source converts pnonc/2 to a native integer
when choosing its starting Poisson index, which cannot represent this value.
The timeout record establishes the observed execution outcome, not a diagnosis
that every large noncentrality call fails in the same way.

Successful wider cases also matter. At df=1e-308, central x=1 retains a
subnormal upper tail near 2.79887e-309. At the same x/df with pnonc=4, the
native pair is near 0.269012/0.730988. An independent df->0 Poisson mixture,
including its zero-degree atom, corroborates this latter limit within the
native absolute accuracy. A legacy port must retain positive tiny df inputs,
not impose the F95 lower bound of 0.001.

The [legacy implementation](dcdflib-nc-chisq.md) now includes 388 implementation
tests and batching measurements.
See the [CDFLIB90 completion audit](cdflib90-completion.md) for the
complete library scope.
