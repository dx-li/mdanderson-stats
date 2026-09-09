# CDFLIB90

CDFLIB90 is a library of cumulative distributions, complementary distributions,
quantiles and inversions with respect to distribution parameters. The catalog
archive contains Fortran 95 version 1.2 and additional C/Fortran DCDFLIB material.
The entry is **partial**. All four public interfaces are implemented for beta,
normal, gamma, chi-square, Poisson, negative-binomial and Student's t distributions. Five other
distribution modules and the remaining archived library interfaces are outstanding.
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
