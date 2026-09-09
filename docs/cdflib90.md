# CDFLIB90

CDFLIB90 is a library of cumulative distributions, complementary distributions,
quantiles and inversions with respect to distribution parameters. The catalog
archive contains Fortran 95 version 1.2 and additional C/Fortran DCDFLIB material.
The entry is **partial**. All four public interfaces are implemented for beta,
binomial, normal, gamma, chi-square, Poisson, negative-binomial, Student's t, F,
noncentral chi-square, noncentral F and noncentral t distributions. All twelve F95 distribution modules are implemented;
the additional legacy and public-support interfaces remain outstanding.
The [106-file inventory](cdflib90-coverage.md) identifies the legacy entry points
and public support APIs that still need contract review and validation.

Source: [catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/21)
and [archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CDFLIB90/CDFLIB90%20%20_V90.tar.gz).
The original [legal terms](../notices/mdanderson-cdflib90-LEGALITIES.txt) are retained.
Original code and compiled reference executables remain outside the package.

## Beta distribution

```python
from mdanderson_stats import cdf_beta, cum_beta, ccum_beta, inv_beta

cdf = cum_beta([0.1, 0.5, 0.9], a=2, b=5)
survival = ccum_beta([0.1, 0.5, 0.9], a=2, b=5)
quantile = inv_beta([0.01, 0.5, 0.99], a=2, b=5)
shape = cdf_beta(3, x=0.2, cum=0.04, b=1).a  # a=2
extreme = cdf_beta(2, ccum=1e-100, a=1, b=1)
# extreme.x rounds to 1; extreme.cx still retains 1e-100.
```

`cdf_beta(which=1, *, cum=None, ccum=None, x=None, cx=None, a=None, b=None)`
returns an immutable `CDFBeta` containing all six broadcast arrays and `which`.
All result arrays own immutable backing storage; they cannot be made writable.
The computed group must be omitted rather than supplied as an initial guess:

| which | Computed group | Required input groups |
|---|---|---|
| 1 (default) | cum and ccum | x/cx, a, b |
| 2 | x and cx | cum/ccum, a, b |
| 3 | a | cum/ccum, x/cx, b |
| 4 | b | cum/ccum, x/cx, a |

Either or both members of each input pair may be supplied. Each lies in [0,1],
and a supplied pair must sum to one within eight double-precision machine
epsilons. The smaller member is retained and only the larger is reconstructed.
This preserves tiny complements supplied alongside a coordinate or probability
that has rounded to one. Omitted complements are calculated from the available
member; information already lost to rounding cannot be recovered.

The convenience functions return the corresponding array. Their positional
arguments are `(x, a, b)` for tails and `(cum, a, b)` for the inverse; use
`x=None, cx=...` or `cum=None, ccum=...` to supply only the complement.
`cdf_beta(which=2)` exposes both quantile coordinates, which is useful at extreme
probabilities. Shapes retain the original inclusive [1e-10,1e10] domain.
The implemented distribution is the standard Beta(a,b), with density proportional
to x**(a-1) (1-x)**(b-1). The manual's printed a/b exponents omit the minus ones;
the actual archived implementation and this port use the standard definition.

## Numerics and failure behavior

Forward evaluation reflects the beta parameters when the supplied complementary
coordinate is smaller. SciPy `betainc` and `betaincc` evaluate both tails directly.
The smaller computed tail is retained and the larger is reconstructed so the
result is a consistent complementary pair. At x=cx=0.5 with a=b, symmetry gives
exactly 0.5 for both tails. This also avoids platform-dependent kernel roundoff
at very large symmetric shapes.
Quantiles invert the smaller probability and compute both coordinates directly
using `betaincinv` and the reflected `betainccinv`; the smaller coordinate is not
obtained by subtracting a rounded larger coordinate from one.

Shape inversion uses monotonicity in a or b and a batched bisection in log shape
over [1e-10,1e10]. Each step evaluates the smaller requested probability using
the complementary-coordinate-aware forward function. Sixty-four steps reduce
the initial log interval to double-precision resolution. This preserves inputs
that a lower-CDF-only shape inverse can lose to subtraction or rounded x.
The result undergoes a forward check with relative probability tolerance 1e-7
plus 32 smallest-subnormal units. Failure raises `ArithmeticError`.

Shape requests require interior coordinate and probability pairs: both members
must be positive, although the larger may round to one. Endpoints cannot identify
a finite shape uniquely and raise `ValueError`. A requested shape outside the
search domain also raises `ValueError`; no boundary is returned as a solution.
Inputs are always validated. There is no unsafe skip-validation option or numeric
status code that callers might accidentally ignore.

These are double-precision calculations. Extremely small tails or quantile
coordinates can underflow to zero, and the larger member can round to one.
Providing the small complement preserves it when representable; it does not
provide arbitrary precision. Shape inversion cannot reconstruct a target that
has already underflowed to zero. Invalid numerical kernel results fail explicitly.

## Validation and source findings

`tools/reference_cdflib_beta.py` compiles eight unmodified original source files
with a separate driver. The fixture records archive/source hashes, compiler,
build command, driver, all inputs, six outputs and the original status for each
of **205 cases**. There are 64 forward cases over varied shapes/coordinates and
141 inverse cases generated from the original forward probabilities.

Forward probabilities agree directly with native output. For inversions, the
original generating coordinate or shape is known and provides a stronger check
than trusting the archived root-finder's output. Python recovers those values
with tight tolerances. Eleven original shape-b inversions return status -50
at the exact initial solution b=5, despite that solution being inside the bounds.
A uniform-beta inverse at lower probability about 0.01 reports status zero but
returns x=0.505 and cx=0.99. These outputs are preserved as source defect evidence,
not relabeled as valid statistical references. Successful native shape inversions
also have noticeably looser accuracy than the Python search.

Independent tests use exact rational binomial sums for integer beta shapes,
closed-form quantiles and shape inversions when one shape is one, endpoints,
logarithmic extreme-tail identities, broadcast batches, shape bounds, invalid
and unidentifiable requests, convenience interfaces and immutable ownership.
The known source status/quantile failures have explicit regression tests.

CDFLIB90's discrete distributions are embedded in continuous beta/gamma formulas;
its inversions may return fractional counts. Their future ports must preserve
that contract, rather than silently substituting integer-valued statistical
quantiles. This beta milestone does not claim those modules are implemented.


## Batching measurements

[Benchmark results](cdflib-beta-benchmark.json) compare one array call with
repeated scalar calls to this same Python API, for forward tails and shape-a
inversion. Both results are checked against one another and analytic beta
identities. The artifact records median-of-three timings, environment versions
and all input-generation parameters. It measures amortized Python overhead and
vectorized numerical work; it is not a comparison against native Fortran.


## Normal distribution

```python
from mdanderson_stats import cdf_normal, cum_normal, ccum_normal, inv_normal

lower = cum_normal([-2, 0, 3], mean=1, sd=2)
upper = ccum_normal([-2, 0, 3], mean=1, sd=2)
quantile = inv_normal(ccum=1e-100, mean=1, sd=2)
location = cdf_normal(3, x=5, cum=0.5, sd=2).mean  # 5
scale = cdf_normal(4, x=3, cum=0.9, mean=1).sd
```

`cdf_normal` computes group 1 (cum/ccum), 2 (x), 3 (mean), or 4 (sd).
The result is an immutable `CDFNormal` containing the computed-group index and
all five broadcast arrays. Omit the computed group; input mean and SD default
to zero and one. Inputs and outputs retain the source's location domain
[-1e100,1e100] and SD domain [1e-10,1e100]. Inversion accepts any positive
representable pair of probability tails, extending the archived 1e-10 cutoff.
Probability-pair consistency and preservation of small complements follow the
beta interface. Zero tails are rejected for inversion because they do not
identify a finite normal quantile.

The standardization is z=(x-mean)/sd. Forward tails reuse the package's validated
`normal_tails` kernel. Inversion computes z from the smaller probability with
SciPy `ndtri`, changing sign for an upper-tail request. It then calculates
x=mean+sd*z, mean=x-sd*z, or sd=(x-mean)/z directly. There is no iterative search
and no loop over individual observations. This retains upper-tail accuracy when
the lower CDF rounds to one. Plain forward probabilities can underflow; callers
needing log tails can use `normal_tails(..., log=True)` with standardized inputs.

SD inversion rejects median probability: z=0 gives either no solution or an
unidentified scale, depending on x and mean. It also rejects negative, zero or
out-of-domain answers. Input bounds remain strict. A computed answer only a few
rounding units beyond a domain endpoint is placed on that endpoint: the maximum
allowance is eight machine epsilons times the absolute endpoint. Larger excursions
raise `ValueError`. This allows a boundary SD to survive CDF/inverse round trips
without accepting genuinely out-of-range input values. Ordinary double-precision
location/scale arithmetic can also lose small offsets relative to large means;
these interfaces do not provide arbitrary-precision reconstruction.

`tools/reference_cdflib_normal.py` compiles eight unmodified original source
files with a separate driver. The fixture records all source/archive hashes,
compiler, command, driver and 243 forward/inverse cases across standardized
positions, means and scales. All five Python outputs are compared with the native
results. Three additional source cases return success while producing a negative,
infinite or NaN SD; tests verify that Python rejects those invalid requests.
Independent tests use the complementary-error-function identity for normal tails,
location/scale inversion, extreme-tail reflection at z=37, defaults, broadcasting,
immutable ownership, strict inputs and computed boundary round trips. The manual's
normal-density illustration omits the negative sign in the exponent; the actual
source and this implementation use the standard exp(-z²/2) density.


## Gamma distribution

```python
from mdanderson_stats import cdf_gamma, cum_gamma, ccum_gamma, inv_gamma

lower = cum_gamma([0.1, 1, 5], shape=2, rate=3)
upper = ccum_gamma([0.1, 1, 5], shape=2, rate=3)
quantile = inv_gamma(None, shape=1, rate=3, ccum=1e-100)
shape = cdf_gamma(3, x=1, rate=2, ccum=0.1353352832366127).shape  # 1
rate = cdf_gamma(4, x=1, shape=1, ccum=0.1353352832366127).rate  # 2
```

`cdf_gamma` computes group 1 (cum/ccum), 2 (x), 3 (shape), or 4 (rate).
Supply the input groups and omit the output group. Input rate defaults to one;
shape has no default. The immutable `CDFGamma` contains `which` and all five
broadcast arrays. Convenience tails accept `(x, shape, rate=1)` and the quantile
accepts `(cum, shape, rate=1, *, ccum=None)`. Pass `cum=None` to supply only ccum.

**The archived argument named SCALE is a rate**, despite a conflicting density
illustration in the source header. Its actual forward routine evaluates the
incomplete gamma at x*SCALE; its inverses divide the unit-rate quantile by SCALE
or by x. Python calls this argument `rate` to make its meaning explicit. The
implemented density is rate**shape * x**(shape-1) * exp(-rate*x) / Gamma(shape).
For a conventional scale parameter theta, supply rate=1/theta.

The original inclusive domains are retained: x in [0,1e100], shape and rate in
[1e-10,1e100]. Probability pairs follow the beta interface, accepting either or
both members and preserving the smaller. Inversion extends the source's upper-tail
1e-10 cutoff to any positive representable upper probability. Lower probability
zero has quantile zero. Upper probability zero has no finite quantile and fails.
Shape/rate inversions require positive x and both probabilities, excluding
unidentified endpoint requests. Out-of-domain solutions raise `ValueError`.
Computed x/rate bounds allow the same eight-epsilon endpoint rounding allowance
as the normal interface; input bounds remain strict.

SciPy `gammainc` and `gammaincc` evaluate the tails directly at x*rate. The smaller
tail is retained and the larger reconstructed for pair consistency. Quantiles
invert the smaller probability using `gammaincinv` or `gammainccinv`, then divide
by rate; rate inversion divides that unit-rate quantile by x. Shape inversion
uses a 64-step batched bisection in log shape over the original domain. It checks
the result against the requested smaller probability with relative tolerance
1e-7 plus 32 smallest-subnormal units. Unattainable brackets raise `ValueError`;
failed numerical kernels or forward verification raise `ArithmeticError`.

Double-precision tails and very small quantiles can underflow to zero. Extremely
large shapes make probability inversion sensitive to tiny shape differences;
a shape answer that cannot meet the forward tolerance fails explicitly. This
interface does not promise arbitrary precision throughout the original domain.

`tools/reference_cdflib_gamma.py` compiles eight unmodified archived Fortran files
with a separate driver and records hashes, compiler, command and **219 cases**:
75 forward evaluations and 48 each of quantile, shape and rate inversion.
Generating parameters are retained, allowing inverse checks even when the native
routine fails. The captured source defects are:

- 36 rate inversions report status 10 because `cdf_gamma(which=4)` treats every
  nonzero `gamma_inverse` status as an error. That inner routine documents positive
  status as successful convergence with an iteration count; quantile inversion
  correctly treats only negative status as failure.
- Nine shape inversions report status -50 at the exact initial solution shape=5.
- Twelve shape-25 forward cases (x*rate equal to 0.01, 0.2, 1 or 5) overstate the
  lower tail by about 0.67%. These are checked against an independent 150-digit
  Decimal evaluation of the integer-shape Poisson-sum identity, not treated as
  correct native references. Other captured forward cases agree directly.

Tests additionally use independent integer-shape identities, exponential tails
and quantiles down to 1e-300, shape/rate recovery, broadcast ownership, empty
arrays, endpoints, strict bounds and invalid or unidentified inversions.
[Gamma batching measurements](cdflib-gamma-benchmark.json) compare array calls
with repeated scalar calls to this same Python API for tails and shape inversion.
Both paths are checked against exponential identities; median-of-three timings
and environment/input details are recorded. These are not Fortran speed ratios.


## Chi-square distribution

```python
from mdanderson_stats import cdf_chisq, cum_chisq, ccum_chisq, inv_chisq

lower = cum_chisq([0.1, 1, 10], df=3)
upper = ccum_chisq([0.1, 1, 10], df=3)
quantile = inv_chisq(None, df=2, ccum=1e-100)
degrees = cdf_chisq(3, x=2, ccum=0.36787944117144233).df  # 2
```

`cdf_chisq` computes group 1 (cum/ccum), 2 (x), or 3 (df). Omit the
computed group and supply the others. Its immutable `CDFChiSquare` contains
`which`, `cum`, `ccum`, `x` and `df`, all arrays broadcast together. Convenience
tails accept `(x, df)`; the quantile accepts `(cum, df, *, ccum=None)`.
Degrees of freedom are real, retaining the source domain [1e-3,1e10]; x retains
[0,1e100]. No default degrees of freedom are assumed.

The source defines chi-square through a gamma distribution with shape=df/2 and
unit-rate coordinate x/2. Python reuses the gamma tails, quantile kernel and
batched shape search. Degrees-of-freedom inversion searches only shapes
[0.0005,5e9] and doubles the answer, preserving the chi-square bounds rather
than relying on a much broader gamma search. The gamma search was extracted
without changing its algorithm so both interfaces use the same numerical logic.

Probability-pair validation, preservation of the smaller tail, quantile zero,
rejection of upper probability zero, and numerical underflow limits follow the
gamma interface. Inverting df requires positive x and both tails. Unattainable
bounds raise `ValueError`; failed forward verification raises `ArithmeticError`.
To preserve boundary roots across small kernel-rounding differences, a requested
smaller probability within 32 machine epsilons relative to an endpoint's tail is
matched to that endpoint. Returned probabilities retain the caller's validated
pair. Computed df permits eight-epsilon relative endpoint rounding; input df
bounds remain strict. Other shape-search answers use the gamma forward tolerance.

`tools/reference_cdflib_chisq.py` compiles nine unmodified source files and records
**80 cases** with archive/source hashes, compiler, command and driver. Thirty
forward cases cover fractional and integer df, including the lower bound; 25
each exercise x and df inversion from the recorded native probabilities.
All captured forward probabilities agree with Python. Forward status values are
not reliable: the source unconditionally finalizes a zero-finder structure that
is unused in the forward path. This build recorded status 50 for all 30 forward
cases; that undefined-state result is not a portable expected status.

Six native x inversions and five df inversions report -50. Some status-zero
inversions also return incorrect answers: at x=5 with generating df=0.001, one
returns df approximately 2.5. Tests recover the known generating parameters
rather than blessing those outputs. Independent tests use the df=1 squared-normal
identity and df=2 exponential identity, extreme tails, boundary round trips,
broadcast ownership, empty arrays and invalid requests. Gamma numerical fixes
also apply to chi-square because they share the same kernels.


## Poisson distribution

```python
from mdanderson_stats import cdf_poisson, cum_poisson, ccum_poisson, inv_poisson

lower = cum_poisson([0, 0.5, 3], mean=2)
upper = ccum_poisson([0, 0.5, 3], mean=2)
count = inv_poisson(0.8, mean=2)  # a real count, not an integer PPF
mean = cdf_poisson(3, s=0, cum=0.1353352832366127).mean  # 2
```

`cdf_poisson` computes group 1 (cum/ccum), 2 (s), or 3 (mean). Supply all input
groups and omit the computed group. Its immutable `CDFPoisson` contains `which`,
`cum`, `ccum`, `s` and `mean`, with arrays broadcast together. The Python `mean`
argument corresponds to the source's LAMBDA. Counts retain [0,1e100] and means
[1e-10,1e100]; neither has a default. Tail conveniences accept `(s, mean)` and the
count inverse accepts `(cum, mean, *, ccum=None)`.

This preserves CDFLIB90's explicitly continuous extension to real counts:
Poisson cum is the upper gamma ratio Q(s+1,mean), and ccum is P(s+1,mean).
At integer s this equals the ordinary inclusive Poisson cumulative probability.
At fractional s it is the gamma extension. `inv_poisson` returns the real solution
of that equation, without integer rounding. It is not the minimum integer count
whose discrete CDF exceeds the requested probability.

Both tails use the shared gamma kernels directly. Count inversion swaps the
probability tails and searches gamma shape over [1,1e100], then subtracts one.
Mean inversion uses the swapped-tail gamma quantile. Complement validation,
small-tail preservation, the 64-step batched log-shape search and its forward
accuracy check follow the gamma interface. Count inversion requires positive
probabilities in both tails. A probability below exp(-mean), apart from endpoint
roundoff, would require a negative count and raises `ValueError`. Probabilities
within 32 machine epsilons relative to the smaller s=0 endpoint tail retain that
boundary root. The original validated pair is returned unchanged.

Computed count/mean bounds allow eight-epsilon relative endpoint roundoff;
input bounds remain strict. Zero/one probabilities do not identify an admissible
positive finite mean in this domain and fail. Plain tails can underflow and
large probabilities round to one. Forming s+1 and subtracting one limits relative
accuracy for very small counts; very large counts can make shape inversion
ill-conditioned. The port retains double-precision arithmetic and fails forward
verification when the shared search cannot meet its documented tolerance.

`tools/reference_cdflib_poisson.py` compiles nine unmodified archived files and
records source/archive hashes, compiler, build command, driver and **61 cases**:
25 forward evaluations plus 18 each of count and mean inversion. Counts include
0.2 as well as integers. Forward tails agree directly; inverse tests recover
known generating parameters. Three native count inversions report -50.
Two additional native cases request probabilities below exp(-mean); both return
success and count zero because the source clamps a negative solution with MAX.
Python rejects these unattainable requests, with explicit regression tests.
Independent tests use 150-digit Decimal Poisson sums, the half-integer gamma
identity at s=0.5, extreme mean inversion, broadcasting, immutable ownership,
empty arrays and strict domain validation.


## Negative-binomial distribution

```python
from mdanderson_stats import cdf_neg_binomial, cum_neg_binomial, ccum_neg_binomial, inv_neg_binomial

lower = cum_neg_binomial([0, 0.5, 3], s=2, pr=0.4)
upper = ccum_neg_binomial([0, 0.5, 3], s=2, pr=0.4)
failures = inv_neg_binomial(0.8, s=2, pr=0.4)
successes = cdf_neg_binomial(3, f=0, pr=0.5, cum=0.25).s  # 2
chance = cdf_neg_binomial(4, f=0, s=1, ccum=1e-100)
# chance.pr rounds to 1; chance.cpr retains 1e-100.
```

`cdf_neg_binomial` computes group 1 (cum/ccum), 2 (f), 3 (s), or 4 (pr/cpr).
Supply all input groups and omit the output group. `CDFNegativeBinomial` contains
`which` and six owned immutable arrays: `cum`, `ccum`, `f`, `s`, `pr`, `cpr`.
They broadcast together. Tail conveniences take `(f, s, pr, *, cpr=None)` and
the failure-count inverse takes `(cum, s, pr, *, ccum=None, cpr=None)`.
Use `None` for a positional probability when supplying only its complement.

The distribution counts failures f before s successes, each trial having success
probability pr and failure probability cpr. Both counts are **real** in [0,1e10],
matching the source parameter table. The CDF is I_pr(s,f+1), the continuous beta
extension of the inclusive discrete negative-binomial CDF. The manual's reference
to failures before the “F'th success” is a typo; the shape is the success count s.
No count inversion rounds to an integer.

Zero required successes are already achieved, so the forward result is cum=1,
ccum=0 for every nonnegative f and any pr, including pr=0. This explicitly defines
the otherwise ambiguous beta corner with zero shape and zero coordinate.
For positive s, pr=0 gives cum=0 and pr=1 gives cum=1. Probability inversion with
positive s supports both probability endpoints. Zero successes cannot identify
pr or a failure count uniquely, so those inverse requests fail.

Count inversion requires positive pr and cpr. Failure inversion also requires
positive s and both probability tails; targets below the f=0 CDF are unattainable.
Success inversion accepts cum=1 as the unique s=0 solution at interior pr; its
other targets require positive tails. Invalid or out-of-range requests raise
`ValueError`. There is no unsafe input-check bypass or ignored numeric status.

The implementation shares the existing beta tail, complementary quantile and
batched log-shape search kernels. The latter two were extracted without changing
the public beta algorithms or bounds. Negative-binomial failure inversion searches
beta shape b in [1,1e10+1] and subtracts one. Success inversion searches positive
shape a from the smallest positive double to 1e10, treating exact zero separately.
This preserves the source's wider count domain, including s below the public
beta interface's 1e-10 minimum and f+1 above its 1e10 maximum.

As in the other CDFLIB interfaces, both complementary inputs can be supplied;
the smaller is retained. Endpoint probability differences within 32 machine
epsilons relative to the endpoint tail retain the corresponding root. Other
count answers undergo the beta search's forward check, with relative tolerance
1e-7 plus 32 smallest-subnormal units. Failed kernels or verification raise
`ArithmeticError`. Tiny quantile coordinates or tails may underflow, and forming
f+1 limits relative accuracy when f is very close to zero. These are ordinary
double-precision calculations, not arbitrary-precision guarantees.

`tools/reference_cdflib_neg_binomial.py` compiles nine unmodified source files
with a separate driver and records archive/source hashes, compiler and command.
The **144 native cases** comprise 36 forward evaluations and 36 each of f, s and
probability inversion, including fractional counts. Forward probabilities agree
directly. The source finalizes an unused zero finder in its forward path; this
build recorded status -50 for all 36 forward cases, which is not a portable
expected status. Nine f inversions and twelve s inversions also report -50.
Inverse validation uses the known generating parameters rather than trusting
these status codes or the archived root-finder's accuracy.

Independent tests use exact rational binomial sums, geometric identities,
pr**s at f=0, success counts down to 1e-300, both count upper bounds, probability
endpoints, tiny complements, mixed zero/nonzero success inversions, broadcasting,
immutable ownership, empty inputs and strict validation.


## Student's t distribution

```python
from mdanderson_stats import cdf_t, cum_t, ccum_t, inv_t

lower = cum_t([-2, 0, 3], df=5)
upper = ccum_t([-2, 0, 3], df=5)
quantile = inv_t(None, df=2, ccum=1e-100)
degrees = cdf_t(3, t=1, cum=0.75).df  # 1: the Cauchy case
```

`cdf_t` computes group 1 (cum/ccum), 2 (t), or 3 (df). Omit the computed
group and supply the others. `CDFStudentT` contains `which` and four owned
immutable broadcast arrays: `cum`, `ccum`, `t`, `df`. Tail conveniences take
`(t, df)`; the quantile takes `(cum, df, *, ccum=None)`. There is no default df.
The source domains are retained: t in [-1e100,1e100], df in [1e-3,1e10].
This is the centered Student's t distribution. The source header's references
to “noncentral t” and its density exponent's missing minus sign are errors;
the actual source uses the centered t-to-beta identity implemented here.

With x=df/(df+t²), cx=t²/(df+t²), the smaller t tail is
I_x(df/2,1/2)/2. Both beta coordinates are formed directly, preserving a small
cx when x rounds to one. The sign of t determines which probability is smaller.
Quantile inversion uses the smaller supplied tail, inverts the corresponding
beta probability and its coordinate complement, then forms sqrt(df)*sqrt(cx/x)
with the appropriate sign. The median gives exactly t=0. Any positive
representable probability pair can be supplied, extending the archived 1e-10
tail cutoff. Zero probability tails have no finite t quantile and fail.

At nonzero fixed t, the smaller tail decreases as df increases. Degrees-of-freedom
inversion uses 64 batched bisections in log df over the original bounds, with
forward verification at relative tail tolerance 1e-7 plus 32 smallest-subnormal
units. Targets within 32 machine epsilons relative to an endpoint tail retain
that endpoint. Computed t and df allow eight-epsilon relative endpoint roundoff;
input bounds remain strict. Out-of-range requests raise `ValueError`, and failed
forward verification raises `ArithmeticError`.

A median probability does not identify df, and a nonmedian target must have the
same sign relative to one-half as t has relative to zero. Numerically identical
tails at both df bounds are also rejected. At large df the distribution approaches
the normal and the inverse can be poorly conditioned; tail accuracy does not imply
the same relative accuracy in df. Tiny tails/coordinates may underflow and extreme
quantiles may exceed the finite source domain. These interfaces use double precision.

`tools/reference_cdflib_t.py` compiles ten unmodified archived files, including
the normal module used by the native initial quantile approximation. It records
source/archive hashes, compiler, command and **70 cases**: 25 forward, 25 quantile
and 20 df inversions. Forward probabilities agree directly. This build returned
status 50 for all forward cases because the source finalizes an unused zero
finder; that undefined-state status is not a portable reference. Five native
quantile cases and four df inversions also report nonzero status. Inverse tests
recover the known generating t/df values rather than trusting those source results.
Independent tests use Cauchy and df=2 identities, t up to magnitude 1e100, tiny
upper probabilities, df boundaries, broadcasting, immutable ownership, empty
arrays and invalid/unidentified requests.


## Binomial distribution

```python
from mdanderson_stats import cdf_binomial, cum_binomial, ccum_binomial, inv_binomial

lower = cum_binomial([0, 0.5, 3], n=5, pr=0.4)
upper = ccum_binomial([0, 0.5, 3], n=5, pr=0.4)
successes = inv_binomial(0.8, n=5, pr=0.4)
trials = cdf_binomial(3, s=0, pr=0.5, cum=0.25).n  # 2
chance = cdf_binomial(4, s=0, n=1, cum=1e-100)
# chance.pr rounds to 1; chance.cpr retains 1e-100.
```

`cdf_binomial` computes group 1 (cum/ccum), 2 (s), 3 (n), or 4 (pr/cpr).
Supply all input groups and omit the computed group. The immutable `CDFBinomial`
contains `which` and six owned broadcast arrays: `cum`, `ccum`, `s`, `n`, `pr`,
`cpr`. Tail conveniences accept `(s, n, pr, *, cpr=None)`; success-count inversion
accepts `(cum, n, pr, *, ccum=None, cpr=None)`. Supply `None` for a positional
probability when providing only its complement.

Counts are real and satisfy 0 <= s <= n <= 1e10. The inclusive binomial CDF is
1-I_pr(s+1,n-s) when s<n, extended to fractional s and n as in the archived
source. At s=n it is one, including zero trials. Inversions return real counts,
without integer rounding. Probability endpoints follow the same distribution:
pr=0 gives cum=1; pr=1 gives cum=0 for s<n and cum=1 at s=n.

Both input pairs preserve their smaller member. Direct complementary beta tails
avoid subtraction when the lower binomial CDF rounds to one. Probability inversion
uses swapped beta tails and returns both pr and cpr directly. It requires s<n;
when s=n, the CDF is independent of pr and cannot identify it uniquely.
The private kernels preserve the full binomial domain, including positive n-s
below the public beta interface's shape minimum.

Success inversion uses a 64-step batched bisection in the fraction s/n over
[0,1], keeping s within [0,n]. Trial inversion solves for beta shape n-s in log
space and reconstructs n. Both check the requested probability against their
count-domain endpoints, allowing at most 32 machine epsilons relative to an
endpoint's smaller tail for rounding. The reconstructed counts undergo a forward
check against the original probability, with relative tolerance 1e-7 plus 32
smallest-subnormal units. This detects precision loss when adding n-s to s.
Unattainable or unidentified requests raise `ValueError`; failed forward
verification raises `ArithmeticError`.

At interior pr, cum=1 uniquely gives s=n for success inversion or n=s for trial
inversion; cum=0 is not a finite admissible solution. At pr=1 the unit-CDF roots
remain unique, while the zero-CDF counts are not. At pr=0, count inversion is
unidentified except when the allowed count domain is a singleton: n=0 for
success inversion or s=1e10 for trial inversion. Those singleton cases require
cum=1. Explicit success/trial bounds and these degenerate cases are validated.

Ordinary double precision limits relative accuracy for counts separated by very
small differences; forming s+1 can also discard tiny success counts. Very small
tails or coordinates can underflow. The forward check prevents silently returning
a reconstructed count that fails the requested tail tolerance.

`tools/reference_cdflib_binomial.py` compiles the primary source and archived
`#cdf_binomial_mod.f90#` backup separately, with nine source files per build.
The backup is renamed to a .f90 filename for compilation without changing its
bytes. Source/archive hashes, both build commands and a common driver are
recorded. Each build supplies **102 cases**: 27 forward evaluations and 25 each
of success, trial and probability inversion. Known generating counts/chances are
retained separately from the actual native inputs. In probability inversion,
pr/cpr start at 0.123/0.877 to expose missing output assignments.

Forward probabilities from both versions agree with Python. Each native build
reports -50 for nine success inversions and six trial inversions. The backup
returns status zero while leaving the placeholder pair unchanged in all 25
probability-inversion cases. Source inspection shows that it moves complement
updates after the termination checks and places final assignments inside just
one branch. The primary version assigns the pair consistently but can still
return an incorrect root with success status; the captured maximum absolute
probability error is 0.2. Python recovers the known generating values rather
than reproducing these defects.

Independent tests use exact rational binomial sums, fractional count round trips,
(1-pr)**n at zero successes down to n=1e-300, count/probability endpoints, the
trial upper bound, tiny complements, broadcasting, immutable ownership, empty
arrays and invalid or unidentified requests. Separate regression checks cover
the backup's unchanged output pair and its forward probabilities.


## F distribution (F95 interface)

```python
from mdanderson_stats import cdf_f, cum_f, ccum_f, inv_f

lower = cum_f([0.1, 1, 10], dfn=5, dfd=10)
upper = ccum_f([0.1, 1, 10], dfn=5, dfd=10)
quantile = inv_f(None, dfn=2, dfd=2, ccum=1e-100)
```

`cdf_f` computes group 1 (cum/ccum) or 2 (f), matching the F95 module and its
parameter table. Omit the computed group and supply both degrees of freedom.
`CDFF` contains `which` and five owned immutable broadcast arrays: `cum`, `ccum`,
`f`, `dfn`, `dfd`. Tail conveniences take `(f, dfn, dfd)`; the quantile takes
`(cum, dfn, dfd, *, ccum=None)`. f retains [0,1e100] and each df retains
[1e-3,1e10]. No default degrees of freedom are assumed.

**The F95 source deliberately omits df inversion.** Its header says which is
restricted to 1:2, its metadata enforces that range, and the actual routine has
only those two branches. A later contradictory comment about returning an
arbitrary df root is stale. The bundled older C/F77 `cdff` supports additional
which=3/4 modes for numerator/denominator df. Those are now available through
the [separate legacy `cdff` interface](dcdflib-f.md), with its wider domains and
independent C/F77 validation. The four F95 interfaces keep their own contract.

The F variable is a ratio of independent scaled chi-square variables. Its beta
coordinate pair is x=dfn*f/(dfd+dfn*f), cx=dfd/(dfd+dfn*f), with shapes dfn/2
and dfd/2. Both coordinates are formed directly, preserving a small coordinate
when its complement rounds to one. The source domains keep the intermediate
product and denominator within floating-point range. The shared beta kernels
produce both tails and invert the smaller supplied probability directly.
The quantile is (dfd/dfn)*(x/cx).

Probability pairs follow the beta interface, extending the archived upper-tail
cutoff to any positive representable ccum. cum=0 gives f=0; ccum=0 has no finite
quantile and raises `ValueError`. A quantile outside the f domain also raises
`ValueError`; computed upper-bound rounding within eight machine epsilons is
accepted at the boundary, while input bounds remain strict. Failed beta kernels
raise `ArithmeticError`. Plain probabilities and coordinates can underflow;
values already rounded to zero/one cannot reconstruct an earlier finite f.

`tools/reference_cdflib_f.py` compiles nine unmodified source files and records
source/archive hashes, compiler, command, driver and **126 cases**: 64 forward
evaluations and 62 quantile inversions. Forward probabilities agree directly in
that grid; inverse tests recover the known generating f. This build records
status 50 for every forward case because the source finalizes an unused zero
finder. Such undefined-state status is not a portable statistical reference.

Two additional tiny-f references at 1e-20 and 1e-100 with dfn=dfd=2 return lower
CDF zero, although the exact answer is f/(1+f). The original coordinate branch
reads xx before assigning it and can then obtain the small coordinate by
subtraction from a rounded one. Python forms both coordinates directly and
retains those tails. Independent tests use F(1,1) as a squared Cauchy variable,
F(2,2)'s rational CDF, reciprocal symmetry with swapped df, equal-df medians,
source-domain endpoints, underflow behavior, broadcasting, immutable ownership,
empty arrays and invalid requests.

## Noncentral chi-square distribution

```python
from mdanderson_stats import cdf_nc_chisq, cum_nc_chisq, ccum_nc_chisq, inv_nc_chisq

lower = cum_nc_chisq([1, 10, 30], df=2, pnonc=4)
upper = ccum_nc_chisq([1, 10, 30], df=2, pnonc=4)
quantile = inv_nc_chisq(None, df=2, pnonc=4, ccum=1e-100)
degrees = cdf_nc_chisq(3, x=10, cum=lower[1], pnonc=4).df
noncentrality = cdf_nc_chisq(4, x=10, cum=lower[1], df=2).pnonc
```

The four public interfaces are `cdf_nc_chisq`, `cum_nc_chisq`, `ccum_nc_chisq`
and `inv_nc_chisq`. The CDF solver computes group 1 (cum/ccum), 2 (x), 3 (df)
or 4 (pnonc). This follows the source's actual branches and parameter metadata;
its introductory numbered list incorrectly omits x and mislabels later modes.
Omit the computed group and supply the other parameters. `CDFNoncentralChiSquare`
contains `which` and owned immutable broadcast arrays `cum`, `ccum`, `x`, `df`
and `pnonc`. The tail conveniences take `(x, df, pnonc)` and the inverse takes
`(cum, df, pnonc, *, ccum=None)`.

Source domains are x in [0,1e100], df in [1e-3,1e10], and pnonc in [0,1e4].
All are explicit inputs except the computed group. Noncentrality is the sum of
**squared** normal means for the sum of squared unit-variance independent normals;
the source prose omits both squares. For real df, the equivalent definition is
a Poisson(pnonc/2) mixture of central chi-square distributions with df+2*j degrees
of freedom. Zero noncentrality reduces to the central distribution. Positive
noncentrality is retained even below the source's 1e-10 central approximation.

Python uses the public [SciPy ncx2 methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html),
verified against the installed 1.18.1 source: CDF/SF evaluate both tails directly;
PPF/ISF invert the smaller supplied probability. The smaller computed tail is
preserved and the larger reconstructed to maintain a complementary pair.
Both df and noncentrality monotonically decrease the CDF for positive x.
Their searches use 64 batched bisections, in log-df or linear noncentrality,
inside the source bounds. Tail matches within 32 machine epsilons preserve
endpoint roots, including pnonc=0. Out-of-bound or numerically indistinguishable
requests fail explicitly. Computed df/x bounds allow eight-epsilon rounding,
following the existing central distribution interfaces; input bounds stay strict.

At x=0 the tails are (0,1). A zero lower probability yields quantile zero;
a zero upper probability has no finite quantile. Parameter inversions require
positive x and both probabilities, because endpoint probabilities cannot identify
the parameter reliably in floating point. Every inverse is checked by forward
evaluation of the smaller tail, with relative tolerance 1e-7 plus 32 smallest
subnormals. A failed consistency check raises `ArithmeticError`.

The original probability validation requires ccum>=1e-10. Python permits smaller
positive tails where the numerical kernels support them, but does not promise
accuracy throughout the extended range. Tests recover lower tails through
1e-300 and upper tails through 1e-100. At df=2, pnonc=4, ccum=1e-300, SciPy 1.18.1
returns a finite inverse whose forward SF differs by a factor greater than ten;
the forward kernel also suffers internal underflow nearby. Python rejects that
inverse. This limit is well outside the archived probability range. A positive
probability whose quantile underflows to zero also fails forward verification.
Forward values alone have the numerical limitations of the underlying kernels;
complement preservation cannot repair their internal underflow.

### Native evidence and independent validation

`tools/reference_cdflib_nc_chisq.py` compiles ten files into two separate profiles,
recording archive/source hashes, compiler, commands, driver and exact patch text.
Each profile has **159 cases**: 48 forward, 40 x, 40 df and 31 noncentrality
inversions. The first profile uses unchanged archive bytes. All its inverse
requests return status 10 because its central chi-square dependency finalizes
an uninitialized root-finder state during forward evaluation. The final status
of the outer forward calls is itself overwritten by unused root-finder state
(status 50 in this build), so it cannot establish validity of those outputs.

The second profile changes only central chi-square status finalization to run
when which>1. No probability or search algorithm is changed. All 111 inverse
requests then return success. The outer forward status remains unreliable.
The native algorithm's series stops at relative term size 1e-5 or total below
1e-20, and forms its upper tail by subtraction. On the reference grid, forward
absolute errors are below 2e-6, but relative errors in small tails can be large:
at x=.2, df=10, pnonc=20, the native lower tail is about 1.32e-26 instead of
4.10e-12; at x=30, df=2, pnonc=.5, its upper tail is about 10% too large.
120-digit independent Poisson mixtures verify both defects. Native inverse
requests based on these approximate tails can move materially from their
known generating parameters. Tests retain and identify four such inverse
comparisons, and verify Python solves the actual requested probability.

Independent tests use 120-digit Poisson mixtures of integer-shape gamma CDFs
for both tails and all inverse modes, the df=1 shifted-normal-square identity,
central reduction, both parameter bounds, tiny positive noncentrality, direct
extreme tails, zero/large x, broadcasting, immutable ownership, empty batches
and invalid/unidentified requests. Original native failures are retained rather
than presented as successful reference answers.

[Batch timings](cdflib-nc-chisq-benchmark.json) compare a broadcast call with
repeated calls to the same Python API, with result agreement checked each time.
On the recorded machine, batches of 64/256 were about 36/105 times faster for
tails and 45/97 times faster for df inversion. These are median-of-three Python
batching measurements, not speedups over native Fortran.

## Noncentral F distribution (F95 interface)

```python
from mdanderson_stats import cdf_nc_f, cum_nc_f, ccum_nc_f, inv_nc_f

lower = cum_nc_f([0.1, 1, 10], dfn=2, dfd=10, pnonc=4)
upper = ccum_nc_f([0.1, 1, 10], dfn=2, dfd=10, pnonc=4)
quantile = inv_nc_f(None, dfn=2, dfd=2, pnonc=4, ccum=1e-80)
noncentrality = cdf_nc_f(3, f=1, dfn=2, dfd=10, cum=lower[1]).pnonc
```

`cdf_nc_f` computes group 1 (cum/ccum), 2 (f) or 3 (pnonc), matching the F95
executable branches and parameter table. Its header lists only two modes, but
the third is implemented in the source. Omit the computed group and supply all
other parameters. `CDFNoncentralF` contains `which` and six immutable owned
broadcast arrays: `cum`, `ccum`, `f`, `dfn`, `dfd`, `pnonc`. Tail conveniences take
`(f, dfn, dfd, pnonc)`; the quantile takes
`(cum, dfn, dfd, pnonc, *, ccum=None)`.

The source domains remain f in [0,1e100], both df in [1e-3,1e10], and pnonc in
[0,1e4]. The distribution is the ratio of a noncentral chi-square divided by dfn
to an independent central chi-square divided by dfd. Noncentrality belongs to
the numerator. Equivalently its CDF is the Poisson(pnonc/2) weighted sum of
I_z(dfn/2+j,dfd/2), with z=dfn*f/(dfd+dfn*f). This retains the original numerator
scale as j changes; simply mixing central F distributions at the same f would
use the wrong scaling.

The separate [legacy `cdffnc` interface](dcdflib-nc-f.md) implements dfn/dfd
inversions at which=3/4 and noncentrality at which=5, retaining wider legacy
input domains and the ignored-q inversion contract. Its unchanged C/F77
fixtures and independent evidence are recorded separately from the F95 port.

Python uses public [SciPy ncf methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncf.html)
for positive noncentrality outside the exact dfd=2 closed form described in
[the legacy notes](dcdflib-nc-f.md), with the installed 1.18.1 source inspected. The
exact central case uses the package's validated F routines for both tails and
quantiles: SciPy 1.18.1 `ncf.sf(1,2,2,0)` returns -0.5. Small positive
noncentralities are retained, rather than applying the archive's <1e-10 central
approximation. Both tails are evaluated directly, the smaller retained and its
complement reconstructed. Invalid kernel probabilities raise `ArithmeticError`.

Quantiles use PPF or ISF according to the smaller probability. A finite result
must satisfy the source coordinate bounds and reproduce the requested smaller
tail. Invalid or inconsistent kernel guesses are refined by 64 batched bisections
in log-f, retaining the candidate with smallest probability error. The initial
reciprocal bracket [1e-100,1e100] is widened to the smallest positive float when
needed. Kernel `OverflowError` invokes this same bounded search; other unexpected
exceptions propagate. Zero lower probability retains f=0 even in a mixed batch
whose other rows require refinement. Zero upper probability has no finite
quantile. Out-of-domain roots fail explicitly.

The refinement is necessary within the archived domain: at dfn=dfd=1e10,
pnonc=1e4, the direct inverse of the CDF at f=1 misses the forward tolerance.
The CDF itself fluctuates slightly at neighboring coordinates, so a last
midpoint is not necessarily the most accurate evaluated candidate. The refined
answer passes forward verification. At dfn=1e10, dfd=2, pnonc=4, the kernel also
reports overflow for a representable quantile near 1e80; the bounded search
recovers it. These checks establish the tested cases, not uniform relative
accuracy for all tails and parameter combinations.

Noncentrality monotonically decreases the CDF at positive f. Its inverse uses
64 batched linear bisections on [0,1e4], with a 32-epsilon smaller-tail allowance
for endpoint roots. Zero f, endpoint probabilities and numerically indistinguishable
boundary probabilities cannot identify this parameter and raise `ValueError`.
All inversions undergo final forward verification at relative tolerance 1e-7
plus 32 smallest subnormals; failures raise `ArithmeticError`. Positive tails
below the original ccum>=1e-10 limit are accepted where kernels and floating
point permit. A finite returned inverse is checked, but forward consistency
alone cannot establish accuracy beyond the forward kernel's own limits.

### Native evidence and independent validation

`tools/reference_cdflib_nc_f.py` compiles eleven archived files into two profiles,
recording their original hashes, compiler, command, driver and any exact patch.
Each profile contains **190 cases**: 72 forward, 68 quantile and 50 noncentrality
inversions. The original profile changes no source bytes. It records status 10
for 18 central quantile requests and all 50 noncentrality inversions: the central
F dependency finalizes unused root-finder state, including at noncentrality zero
when establishing a search bound.

The second profile guards only that central F status finalization with which>1.
No probability or inversion algorithm changes. All 118 inverse requests then
return status zero. The outer noncentral F forward finalization still operates
on unused root-finder state (status 50 in this build); that status is not a
portable validity check. Native forward values on this grid agree within the
source summation's 1e-4 absolute scale, but the series' additional early stop
when its accumulated sum is below 1e-20 can discard almost the entire result.
At f=.1, dfn=.5, dfd=10, pnonc=20, it returns about 3.31e-22 instead of 2.38e-5.
A 120-digit independent beta mixture verifies this defect. Another recorded
native probability shifts its noncentrality inverse from generating value 4
to approximately 4.01438 under the accurate CDF; tests identify that discrepancy
and verify the actual input probability is solved.

Independent tests use 120-digit Poisson mixtures with finite beta identities,
including fractional numerator df. When dfd=2, the exact CDF is
z**(dfn/2)*exp(-(pnonc/2)*(1-z)); log1p/expm1 forms provide independent small-tail
checks for df up to 1e10 and coordinates through 1e80. Other tests cover zero
noncentrality in mixed arrays, small positive noncentrality, both df bounds,
noncentrality 1e4, zero f, f=1e100, overflow refinement, immutable ownership,
broadcasting, empty batches and invalid/unidentified requests.

[Batch timings](cdflib-nc-f-benchmark.json) compare one broadcast call against
repeated scalar calls to the same Python API, with agreement checked each time.
Batches of 64/256 were about 35/130 times faster for tails and 49/116 times faster
for noncentrality inversion on the recorded machine. These median-of-three
measurements describe Python batching, not speedup over native Fortran.

## Noncentral t distribution

```python
from mdanderson_stats import cdf_nc_t, cum_nc_t, ccum_nc_t, inv_nc_t

lower = cum_nc_t([-2, 0, 2], df=2, pnonc=0.5)
upper = ccum_nc_t([-2, 0, 2], df=2, pnonc=0.5)
quantile = inv_nc_t(0.95, df=10, pnonc=2)
noncentrality = cdf_nc_t(4, t=2, df=2, cum=lower[2]).pnonc
roots = cdf_nc_t(3, t=1, pnonc=3, cum=0.03, df_bracket=([0.001, 0.2], [0.2, 100])).df
```

`cdf_nc_t` computes group 1 (cum/ccum), 2 (t), 3 (df) or 4 (pnonc). Omit the
computed group and supply the other parameters. `CDFNoncentralT` contains
`which` and five owned immutable broadcast arrays: `cum`, `ccum`, `t`, `df`,
`pnonc`. Tail conveniences take `(t, df, pnonc)` and the inverse takes
`(cum, df, pnonc, *, ccum=None)`. The df inverse additionally accepts
`df_bracket=(lower, upper)`, whose endpoints broadcast with the other inputs.
This option is rejected for other modes.

The source domains remain t in [-1e100,1e100], df in [1e-3,1e10], and pnonc in
[0,1e4]. Noncentral t is (Z+pnonc)/sqrt(V/df), where Z is standard normal and
V is independent chi-square(df). Unlike noncentral F/chi-square, pnonc here is
a **normal mean**, not a sum of squared means. Although the mathematical
distribution supports negative noncentrality, this F95 interface retains its
nonnegative source bound; remaining legacy contracts are reviewed separately.

### Root selection and numerical behavior

The df CDF is not generally monotone. For t=1, pnonc=3 and CDF=.03, there are
two roots near .03926845456 and 1.65497762959. The example above selects both
using separate brackets. The source's default interval [.001,1e10] has
same-sign endpoint residuals and misses both. Python retains that default
interval and raises a clear error asking for `df_bracket` when the root is not
bracketed. It does not interpret this failure as proof that no df solution exists.

A supplied bracket must have distinct ordered endpoints inside the source df
bounds and must straddle the desired probability, or match an endpoint. The
solver performs 64 bisections in log-df while preserving residual signs, without
assuming monotonicity. It returns one bracketed root, not an enumeration of all
roots. A same-sign bracket containing two crossings or a tangent root needs a
more appropriate interval; this contract is explicit rather than claiming that
finite sampling can find every root. At t=0, the CDF is Phi(-pnonc), independent
of df, so df inversion is unidentified and rejected.

Noncentrality monotonically decreases the CDF and is inverted on [0,1e4] with
64 linear bisections. Endpoint matches within 32 machine epsilons of the smaller
tail preserve boundary roots. Resolved endpoint rows are excluded from further
search, including in mixed batches. Numerically identical endpoint probabilities
cannot identify a parameter and fail explicitly. All inverses require positive
cum/ccum and undergo smaller-tail forward verification at relative tolerance
1e-7 plus 32 smallest subnormals. Failed verification raises `ArithmeticError`.
Computed t/df bounds use the existing eight-epsilon rounding allowance; input
bounds remain strict.

Python uses public [SciPy nct methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html),
with the installed 1.18.1 source inspected, for the general CDF/SF and PPF/ISF
calculations. Zero noncentrality uses the package's central t implementation;
zero t uses its normal tails. The smaller computed probability is retained and
the larger reconstructed. The original probability cutoff is 1e-10 on both
tails. Python accepts smaller positive probabilities where numerical kernels
permit, with no claim of uniform accuracy in the extended range. At df=2 and
t=pnonc=1e4, the underlying kernel's quantile differs from the generating t by
about 1.5e-8 relatively while satisfying the probability tolerance; independent
df=2 checks use an appropriate 1e-7 tolerance at that large noncentrality.

### Small negative-tail repair

For t<0 and positive noncentrality, the direct kernel can suffer cancellation
or return NaN, including at ordinary trial parameters during a noncentrality
search. If its lower tail is below 1e-6 or nonfinite, Python instead conditions
on the normal numerator:

P(T<=t) = integral from 0 to infinity of
phi(pnonc+u) * P_gamma(df/2, df*u**2/(2*t**2)) du.

The integrand is nonnegative and avoids subtracting nearly equal tails. The
normal density at pnonc is factored out, and s=(pnonc+1)*u resolves its shrinking
scale. Adaptive quadrature uses zero absolute tolerance, relative tolerance
2e-11 and at most 200 subdivisions. An integration failure, nonfinite result or
relative error estimate above 1e-8 raises `ArithmeticError`. When Phi(-pnonc)
already underflows to zero, it is a decisive upper bound on this negative tail.
These difficult rows use scalar quadrature; ordinary rows retain batched kernels.

Tests independently condition on the chi-square denominator, integrating
Phi(t*r-pnonc) against the known radial density using two converged Simpson
grids. This is a different integral and numerical method from the fallback.
It verifies both a cancellation case and a former NaN at t=-5, df=2,
pnonc=9.765625. Other independent checks use the closed-form df=2 distribution
and the exact normal identity at zero t, including noncentrality 37.

### Native evidence and performance

`tools/reference_cdflib_nc_t.py` compiles eleven source files into two profiles,
recording original hashes, compiler, command, driver and exact patch text.
Each profile retains **204 cases** (60 tails, 57 t, 45 df and 42 noncentrality
requests), plus the explicit two-root df example. The unchanged original returns
status 10 for 15 quantile and 12 df requests through its central t dependency.
A separate profile guards only that dependency's unused root-finder status
finalization with which>1; it makes no probability/search changes. The outer
noncentral t forward status is still derived from unused solver state (50 here).

The status-repaired profile returns success for 45 quantile, 42 df and all 42
noncentrality requests. Twelve quantile requests still return -50 and three df
requests return 50; the explicit two-root example also fails its full-bound
search. Successful status alone does not establish an accurate answer. Most
native forward differences on the fixture grid are below 1e-8, but at t=.5,
df=10, pnonc=4 the source treats intermediate beta complements below 1e-10 as
proof that t is effectively zero. It returns Phi(-4), about 3.17e-5, instead of
about 2.41e-4. Independent denominator integration verifies this defect.
Tests retain the resulting displaced/unattainable inverse requests explicitly.
Native df comparisons use narrow explicit brackets around their generating df
to isolate that root rather than enclosing both crossings.

Validation also covers both df bounds, noncentrality zero/1e4, central reduction,
multiple-root selection and sign changes, mixed endpoint batches, immutable
ownership, empty arrays and invalid brackets/inputs. All four public F95
interfaces are implemented, but the catalog entry remains partial pending the
[legacy and supporting contracts](cdflib90-coverage.md).

[Batch timings](cdflib-nc-t-benchmark.json) compare one broadcast call to repeated
scalar calls of the same Python API, checking agreement each time. For batches
of 32/128, positive-t tails were about 21/65 times faster and noncentrality
inversions about 25/62 times faster. Quadrature-heavy negative-tail batches were
about 1.30/1.34 times faster. These median-of-three measurements describe Python
batching, not speedup over native Fortran, and expose the fallback's cost.


The separate [legacy normal interface](dcdflib-normal.md), `cdfnor`/`cumnor`,
implements all four C/F77 normal modes without the F95 location/scale bounds.
It adds scaled arithmetic for intermediate overflow, subnormal tail recovery
and forward verification, with unchanged native C and F77 reference fixtures.


The [legacy Student t interface](dcdflib-t.md), `cdft`/`cumt`, additionally
supports unrestricted finite t and positive df inputs, with the C/F77 df search
range [1e-100,1e10], logarithmic overflow repair and separate native validation.

`cdfgam` and `cumgam` implement the [legacy gamma interface](dcdflib-gamma.md),
including all four computed groups, explicit rate semantics, wide finite domains,
logarithmic scaling and tiny-shape tail repairs validated against independent
high-precision calculations and unchanged C/F77 references.

`cdfchi` and `cumchi` implement the [legacy chi-square interface](dcdflib-chisq.md),
with wide finite inputs, bounded x/df inversions, subnormal rounding repairs and
unchanged C/F77 references checked against independent high-precision identities.

`cdfpoi` and `cumpoi` implement the [legacy Poisson interface](dcdflib-poisson.md),
including zero mean, wide finite inputs, bounded continuous count/mean inversions
and independent repairs for native overflow and false-success results.

`cdfnbn` and `cumnbn` implement the [legacy negative-binomial interface](dcdflib-neg-binomial.md),
with all four computation modes, wider counts, paired chance inversions and
independently validated repairs for extreme shapes and small inverse targets.

`cdfbin` and `cumbin` implement the [legacy binomial interface](dcdflib-binomial.md),
with separate success/trial search bounds, complementary chance inversion and
independent checks of native process failures, tiny counts and wide inputs.

`cdfbet` and `cumbet` implement the [legacy beta interface](dcdflib-beta.md),
with wide positive shapes, bounded shape inversions, complementary quantiles
and positive recurrences that preserve tails when both shapes are tiny.
