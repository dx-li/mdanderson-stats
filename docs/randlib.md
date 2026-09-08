# RANDLIB

Catalog entry 27 is partial. The 32-stream generator bank and state controls are
implemented, along with bounded uniforms, permutations, exponential, normal, gamma,
central/noncentral chi-square, F, beta, binomial, Poisson, negative-binomial,
multinomial and multivariate-normal sampling. The final archive
coverage/performance audit remains pending.

The [official entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/27)
lists version 90, modified September 27, 2002. RANDLIB_V90.tar.gz contains
Fortran 77, C and Fortran 95 implementations and their documentation. The
Fortran 77 readme identifies version 1.3, August 1997. Original archives, source
and reference executables remain local research material. Retained legal terms
and attribution are linked from THIRD_PARTY_NOTICES.md.

## Explicit generator banks

```python
from mdanderson_stats import RandlibGenerator, ranlist_seeds

bank = RandlibGenerator(seed=ranlist_seeds("example"))
raw = bank.integers(1000)
u = bank.uniform(1000)
bank.select(2)
bank.set_antithetic(True)
paired = bank.uniform(1000)
bank.reinitialize(1)  # First draw of the next 2**30-draw block is next
current_pair = bank.get_seeds()
```

Each `RandlibGenerator` owns 32 streams with independent current, initial and
block-start states and antithetic flags. Stream starts are separated by 2^50
draws. Default seeds are (1234567890, 123456789); explicit seeds must be integers
in 1..2147483562 and 1..2147483398. The recurrence and vectorized modular-power
helper are shared with the already validated RANLIST implementation.

`integers(size=1)` returns a read-only vector of consecutive raw IGNLGI draws
in 1..2147483562, advancing the selected stream exactly that many times.
`uniform(size=1)` divides those integers by 2147483563 in double precision.
`legacy=True` instead applies RANF's single-precision integer conversion and
scale constant, returning the resulting float32 values as float64. Empty draws
leave state unchanged. `max_draws=1_000_000` bounds a batch before allocation;
a different positive limit may be supplied to the constructor. Invalid requests
do not change state. A bank is mutable; concurrent callers need coordination
or separate banks. Separate objects share no mutable RNG state.

| Method | Original routine | Effect |
| --- | --- | --- |
| `select(stream)` / `stream` | SETCGN / GETCGN | Select or inspect stream 1..32 without restarting it |
| `get_seeds()` | GETSD | Return current component states of selected stream |
| `set_seeds(pair)` | SETSD | Replace selected stream's initial, block and current states |
| `set_all_seeds(pair)` | SETALL | Rebuild all streams from the supplied stream-one seed, retaining selection and antithetic flags |
| `set_antithetic(enabled)` | SETANT | Complement future raw draws as M1-z; internal component states are unchanged |
| `reinitialize(-1)` | INITGN(-1) | Reset block and current states to selected stream's initial state |
| `reinitialize(0)` | INITGN(0) | Return to current block's start |
| `reinitialize(1)` | INITGN(1) | Move block start forward by 2^30 draws, then reset current state to that start |
| `advance_state(k)` | ADVNST(k) | Advance current state by 2^k draws and adopt it as the new initial/block/current state |

Block advancement is measured from the block start, not from the last consumed
draw. ADVNST differs: it advances from the current state and replaces the reset
anchor. Exponents are nonnegative integers below 2^53. Prime-modulus exponent
reduction makes even large advances practical without constructing 2^k or
replaying the intervening sequence. The raw stream recurrence and state remain
identical regardless of antithetic mode. Modern ordinary/antithetic uniforms
sum to one to floating-point precision; legacy float32 rounding need not obey
that identity exactly.

The archived Fortran SETALL implementations fail to select stream one before
its reset. Calling them while another stream is selected can leave stream one's
current/block state stale. Python initializes every stream explicitly, so the
selected stream cannot affect seed initialization. Tests cover this correction.
Native comparison sequences reseed all streams while stream one is selected.
Invalid stream zero, invalid seeds and negative advance exponents are rejected
rather than preserving unsafe or accidental source behavior.

`get_seeds` is a snapshot of the selected current pair, not a serialized bank;
restoring it with `set_seeds` intentionally replaces the reset anchors. Full bank
serialization and the remaining non-uniform samplers are not yet implemented.

## Validation

`tools/reference_randlib_streams.py` compiles unchanged Fortran 77 base routines
and the Fortran 95 ecuyer_cote_mod independently. Sixty-four cases cover two
seed pairs, four streams and advance exponents 0/5/30/80. Each records fourteen
integer draws and their current component states through antithetic changes,
block resets, advancement, seed replacement and stream switching. Python matches
every value exactly. Fixtures include source hashes, compiler and flags.

Independent tests check split versus combined batches, interleaved streams,
antithetic complements, block-start semantics, ADVNST anchor replacement,
prime-modulus assumptions, large powers, read-only outputs and invalid-operation
state preservation. These comparisons validate the bank foundation. The bounded samplers below
have additional native checks; exponential sampling is validated separately below.


## Bounded uniforms and permutations

```python
bank = RandlibGenerator()
continuous = bank.uniform(100, low=-2.3, high=9.7)
discrete = bank.integer_uniform(-3, 7, 100)
shuffled = bank.permutation([10, 20, 30, 40])
source_fortran = bank.uniform(100, low=-2.3, high=9.7, legacy=True)
source_c = bank.uniform(100, low=-2.3, high=9.7, legacy=True, source="c")
```

`uniform` now accepts finite scalar bounds. Default arithmetic avoids overflow
from subtracting opposite extreme bounds and returns exactly the requested
constant for equal bounds. Equal real bounds still consume the requested draws,
matching GENUNF. Floating-point rounding may reach a bound. Legacy requires the
bounds and their difference to fit float32 and follows the archived arithmetic.
`source="fortran"` (default) rounds the raw integer and scale constant to float32
before multiplication. `source="c"` uses the C source's double-precision scaling
then rounds to float32; it requires `legacy=True`. Both subsequently perform
the bounded affine transformation in float32. C's rounding can produce an
exact uniform of one, despite the original endpoint-exclusion comment. Modern
unit uniforms stay strictly below one. Compiler contraction is disabled in C
reference checks so intermediate arithmetic is explicit.

`integer_uniform(low, high, size=1, legacy=False, max_attempts=None)` returns
inclusive integer values. Bounds are signed 32-bit integers and the interval
width must be at most 2,147,483,562, the raw generator's range. Equal integer
bounds consume no draws. Default rejection sampling accepts an exact multiple
of the interval width, eliminating modulo bias. Legacy uses IGNUIN's inclusive
rejection endpoint, which gives an extra accepted residue zero. At the full raw
range width, that source rule accepts only zero; it can take an impractical
number of draws. A finite attempt budget prevents unbounded execution.

`permutation(values, legacy=False, max_attempts=None)` copies a one-dimensional
signed-32-bit integer array and performs GENPRM's forward Fisher–Yates swaps.
Repeated and negative values are allowed; empty and singleton inputs consume
no draws. Legacy selects swap positions with the archived IGNUIN rule. Modern
uses unbiased positions. The input is unchanged and the returned array is
read-only. The bank's `max_draws` also limits permutation length.

Integer/permutation attempt budgets default to `max(100_000, 4 * output_size)`.
An explicit positive `max_attempts` overrides this. The budget counts raw draws,
including accepted draws, and covers the entire operation. Exceeding it raises
`ArithmeticError` and leaves the bank state unchanged. Batched rejection sampling
never consumes unused trailing draws: each chunk contains at most the number of
outputs still needed. Small tails and permutation swaps use scalar integer
recurrences to avoid allocating a NumPy batch for every single draw.

`tools/reference_randlib_sampling.py` compiles unchanged C, Fortran 77 and
Fortran 95 routines. Its 144 cases compare bounded integer draws, real uniforms,
permutations and their resulting component states across two seed pairs, three
streams, both antithetic settings and four integer ranges. These include forced
rejections, constant bounds and a wide range with roughly half of draws rejected.
All values and states match their selected source arithmetic. Independent tests
cover unbiased rejection, exact batch/scalar state agreement, full-range behavior,
transactional budget failures, multiset preservation and extreme real bounds.


## Exponential sampling

```python
bank = RandlibGenerator()
values = bank.exponential(10000, mean=2.5)
source_values = bank.exponential(100, mean=2.5, legacy=True)
c_values = bank.exponential(100, mean=2.5, legacy=True, source="c")
```

`exponential(size=1, mean=1, legacy=False, source="fortran", max_attempts=None)`
implements GENEXP, with mean one corresponding to SEXPO. Means must be finite
and nonnegative. Zero is the original routine's degenerate case and still
consumes random draws. Default uses a vectorized `-log(U) * mean` transformation
with one double-precision unit uniform per result. It follows the selected
stream's antithetic setting and has the finite resolution of the base generator.
The default targets the exponential distribution with this finite-resolution
uniform stream. Its sample sequence differs from SEXPO.

Legacy implements the archived Ahrens–Dieter algorithm SA with its float32 table,
doubling steps, running minima and arithmetic. It can consume multiple uniforms
per sample; `source="c"` uses C's distinct uniform conversion. Mean-one and scaled
results follow the same path, including zero mean. The source's exact-0.5
uniform branch returns zero, following the corrected strict comparison in all
three archived implementations. Legacy means must be representable in float32;
positive means that underflow to zero are rejected.

The attempt budget defaults to `max(100_000, 4 * size)` and counts raw draws.
Budget failures, overflow and invalid inputs leave bank state unchanged.
Returned arrays are read-only. Mean multiplication may underflow small individual
samples to zero at the selected precision; overflow raises `ArithmeticError`.
C's unit-uniform rounding can yield one, which would overrun SEXPO's eight-entry
table. Python detects that case and raises `ArithmeticError` without committing
state, rather than reading past the table. Fortran's maximum uniform follows the
valid final table branch.

`tools/reference_randlib_exponential.py` compiles unchanged C, Fortran 77 and
Fortran 95 routines. Its 110 cases record 20 values and corresponding component
states each, covering mean zero/one/2.3, three streams, antithetic mode, an exact
half-uniform seed and the Fortran final-table branch. Values and states match
exactly in the selected legacy mode. Batch calls match repeated scalar calls.
Independent tests verify the modern inverse transform, mean/variance/CDF on
50,000 draws, degenerate means, source boundary behavior, limits and rollback.

## Normal sampling

```python
bank = RandlibGenerator()
z = bank.normal(1000)  # mean 0, standard deviation 1
x = bank.normal(1000, mean=-2.3, sd=3.7)
original = bank.normal(1000, legacy=True)  # Fortran 77 / 95 SNORM + GENNOR
original_c = bank.normal(1000, legacy=True, source="c")
```

`normal(size=1, *, mean=0, sd=1, legacy=False, source="fortran",
max_attempts=None)` returns a read-only float64 vector. Mean must be finite;
standard deviation must be finite and nonnegative. Empty requests consume
nothing. A zero standard deviation returns the mean but still consumes the
same draws as a nondegenerate request in that mode.

The default applies SciPy's standard-normal inverse CDF to batched raw uniforms,
then scales and shifts the result. It consumes exactly one raw draw per result,
with identical scalar/batch sequences and no hidden cached spare variate.
The finite uniform grid limits the accessible quantiles; it is not an
infinite-resolution continuous normal generator. Its sequence differs from FL.

Legacy mode implements the archive's Ahrens–Dieter FL (M=5) center/tail rejection
algorithm in single precision, including variable draw consumption. C and
Fortran have different printed table constants, uniform conversion and threshold
arithmetic. `source="c"` selects those C details; it requires `legacy=True`.
Results are widened to float64 only after source rounding. Legacy parameters
must fit float32 without rounding a nonzero parameter to zero. Individual small
results may underflow during source arithmetic.

`max_attempts` bounds raw draws across the entire request, defaulting to
`max(100000, 4*size)`. Budget exhaustion, table-bound failure or nonfinite
arithmetic raises `ArithmeticError` without committing any generator state.
Overflow is reported even if a different ordering of scale/shift arithmetic
could have avoided an intermediate overflow. Invalid parameters raise
`ValueError` before sampling. Other streams and reset anchors are untouched.

`tools/reference_randlib_normal.py` compiles the unchanged original C, Fortran
77 and Fortran 95 implementations with recorded compiler flags and source
hashes. Its fixture contains 450 cases / 9,000 values and component states:
ordinary seeds, all 31 central intervals, both tails, exact-half and extreme
uniforms, antithetic flags, selected streams and zero/nontrivial scale and mean.
Tests compare every native value and state exactly, and additionally check
scalar/batch equivalence, inverse-CDF round trips, antithetic symmetry, empirical
moments/CDF, parameter validation and failed-request rollback.

`tools/benchmark_randlib.py` measures 10,000 normal, exponential and gamma draws in
one batch versus repeated scalar calls to the same default-mode Python API.
The environment and measured times are in [randlib-benchmark.json](randlib-benchmark.json).
This comparison measures Python batching, not speed relative to original native
programs or legacy rejection sampling. RANDLIB's remaining distributions and
final whole-archive audit are still pending.

## Gamma sampling

```python
bank = RandlibGenerator()
x = bank.gamma(1000, shape=2.5, rate=1.7)
original = bank.gamma(1000, shape=2.5, rate=1.7, legacy=True)
original_c = bank.gamma(1000, shape=2.5, rate=1.7, legacy=True, source="c")
```

`gamma(size=1, *, shape=1, rate=1, legacy=False, source="fortran",
max_attempts=None)` samples the density
`rate**shape * x**(shape-1) * exp(-rate*x) / Gamma(shape)`.
Both parameters must be finite and strictly positive. Mean is `shape/rate`
and variance is `shape/rate**2`. RANDLIB calls its first argument “location”
(and the Fortran 95 argument is named `scale`), but its density and division
by that argument make it a **rate**. Python names the parameter accordingly.

Default mode uses SciPy's vectorized inverse regularized incomplete gamma
function, followed by division by rate. It consumes one raw uniform per output.
The finite uniform grid and floating-point underflow limit attainable values;
very small shapes may produce zero. It has a different sample sequence from
legacy GS/GD, and shape 1 is not required to match the sequence of `exponential`.

Legacy mode uses GS below shape 1 and GD at or above 1, including the original
normal/exponential primitives, coefficient tables, squeeze/quotient/hat tests
and the large-quotient overflow correction. C and Fortran constant promotion,
transcendental precision and comparison thresholds are selected explicitly.
Logarithms use double-precision evaluation followed by rounding to the selected
source precision. This avoids CPU-dependent float32 logarithm approximations
whose one-unit rounding differences GS amplifies for small shapes.
Shape-dependent constants are local to a sampling request, preserving behavior
when callers alternate between shapes without sharing process-global caches.
Legacy parameters must remain positive and finite after float32 conversion.

Results are read-only float64 arrays (legacy results are first calculated in
float32). Zero-size calls do not consume draws. Underflow to zero is allowed;
nonfinite outputs or nonfinite legacy quotients raise `ArithmeticError`.
`max_attempts`, default `max(100000, 4*size)`, bounds the total number of raw draws
including rejection trials and nested normal/exponential calls. Every failure
leaves the bank unchanged. Source exponential-table guards also apply when
GS/GD calls that primitive. Invalid arguments raise `ValueError` before sampling.

`tools/reference_randlib_gamma.py` compiles unchanged C, Fortran 77 and
Fortran 95 code and records hashes, compiler versions and flags. Its fixture
has 180 cases / 3,600 values and states, covering shapes below/at/above 1,
3.686 and 13.022 regime boundaries, small and large shapes, nonunit rates,
multiple streams/seeds, antithetic draws, mixed 5 / 0.5 / 5 shape sequences,
and a targeted large-quotient GD acceptance case.
Native state comparisons are exact. Value comparisons allow relative error
`4e-6` plus two float32 subnormal units for cross-platform transcendental
rounding; on the reference-generation machine these values matched exactly.
Additional tests check empirical moments/CDFs, inverse-CDF round trips,
scalar/batch equivalence, underflow, invalid arguments and rollback.

The batching benchmark includes 10,000 gamma draws with shape 2.5 and rate 1.7;
its measured times and comparison limits are recorded in
[randlib-benchmark.json](randlib-benchmark.json). Remaining count and multivariate distributions and RANDLIB's final coverage audit
remain pending.

## Chi-square and F sampling

```python
bank = RandlibGenerator()
central = bank.chi_square(1000, df=5)
noncentral = bank.noncentral_chi_square(1000, df=5, noncentrality=2.3)
ratio = bank.f(1000, dfn=5, dfd=12)
noncentral_ratio = bank.noncentral_f(1000, dfn=5, dfd=12, noncentrality=2.3)
original = bank.noncentral_f(1000, dfn=5, dfd=12, noncentrality=2.3, legacy=True)
```

All four methods accept `size=1`, `legacy=False`, `source="fortran"` and
`max_attempts=None`. Degrees of freedom default to 1 and noncentrality defaults
to 0. Degrees of freedom must be finite and positive; noncentrality must be
finite and nonnegative. Legacy noncentral methods additionally require the
numerator degrees of freedom to be at least 1. Default noncentral methods
support all positive degrees of freedom. `source="c"` requires `legacy=True`.
Results are read-only float64 vectors. Empty batches do not advance the stream.

Defaults evaluate SciPy quantiles on a vector of raw uniforms, consuming one
draw per output. Noncentrality zero explicitly uses the central distribution,
so the corresponding default central and noncentral sequences are identical.
Positive results are checked against the smaller CDF/survival tail with absolute
tolerance `64*float64_epsilon + 1e-7*min(u, 1-u)`. This catches extreme F quantiles
that a library returns as a finite limiting value even though their implied
probability is incorrect. Nonfinite/negative results or failed probability
checks raise `ArithmeticError` and leave state unchanged. Finite-resolution
uniforms limit attainable tails; underflow to zero remains allowed.

Legacy methods compose the original single-precision gamma and normal samplers.
Central chi-square doubles a gamma draw with shape `df/2`. Central F draws
numerator and denominator gamma components in that order, normalizes by their
degrees of freedom, then divides. Noncentral chi-square adds a gamma component
with shape `(df-1)/2` to a squared normal shifted by `sqrt(noncentrality)`.
Noncentral F uses that numerator and a central denominator.

The reference gfortran builds evaluate the normal term before the gamma term
in noncentral expressions; the reference C build evaluates gamma first.
Legacy modes preserve those recorded orders, including in batch calls.
Fortran does not guarantee expression evaluation order across compilers;
these sequence comparisons refer to the tool's recorded builds, not every
possible native compiler/optimization setting.

The archived noncentral routines use the normal-only branch when the rounded
degrees of freedom are below `1.000001`, ignoring the small residual gamma
component. GENNF also omits division by the numerator degrees of freedom in
that branch. The threshold itself is single precision in Fortran and double
precision in C, so their behavior differs at a representable boundary near 1.
Default mode uses the actual parameter values without this approximation.

Legacy F routines return float32 `1e37` when the normalized denominator is no
greater than `1e-37` times the numerator, including the original zero/zero case.
Python emits one `RuntimeWarning` per affected request, rather than printing
one message per value. Those intentionally capped results consume the native
draws. Treating the warning as an exception prevents state commitment.
Default mode does not apply that cap. Other nonfinite legacy results raise an
error. Legacy parameters and derived gamma shapes must remain representable
and valid in float32; silent parameter underflow is rejected.

The total raw-draw budget defaults to `max(100000, 8*size)` and includes nested
gamma, normal and exponential rejection draws. Exhaustion rolls back the entire
request. Selected-stream initial/block states and other streams are untouched.
The source-compatible exponential endpoint guard remains in force when called
by a nested sampler.

`tools/reference_randlib_chi_f.py` compiles the unchanged C, Fortran 77 and
Fortran 95 routines and records source hashes, compiler versions and flags.
Its fixture contains 552 cases / 11,040 values and component states, covering
all four routines, two streams, antithetic draws, fractional/integer and near-one
degrees of freedom, zero/positive noncentrality and intentional F truncation.
Tests require exact states and source warning behavior, with value tolerance
`5e-6` relative plus two float32 subnormal units for transcendental rounding.
They also check scalar/batch identity, moments and CDFs, default noncentral
parameters below one, zero-noncentrality identity, invalid arguments, probability
validation and transactional failures. The batching benchmark includes all four
methods; its comparison remains against repeated calls to the same Python API.

Remaining count and multivariate samplers and the final RANDLIB coverage/performance
audit remain pending.

## Beta sampling

```python
bank = RandlibGenerator()
x = bank.beta(1000, a=2, b=3)
original = bank.beta(1000, a=2, b=3, legacy=True)
original_c = bank.beta(1000, a=2, b=3, legacy=True, source="c")
```

`beta(size=1, *, a=1, b=1, legacy=False, source="fortran",
max_attempts=None)` returns a read-only float64 vector. Both shapes must be
finite and positive. Mean is `a/(a+b)` and variance is
`a*b/((a+b)**2*(a+b+1))`. The default uses SciPy's vectorized inverse
regularized incomplete beta function, with one raw uniform per output.
The uniform special case `a=b=1` reproduces default uniform draws. Zero-size
requests do not consume draws. Floating-point rounding can produce 0 or 1,
especially for tiny shapes; no artificial clipping moves those endpoints.

Legacy mode implements GENBET's BB algorithm when both rounded shapes exceed
1 and BC otherwise. It preserves shape ordering, rejection tests, coefficient
rounding, two-uniform trials and the source's exponential and logarithm guards.
Shape-dependent state belongs to the request, so alternating or swapping
shapes cannot reuse stale constants. `source="c"` requires `legacy=True` and
selects the C expression promotions and squared-uniform calculation. Logarithm
and exponential evaluation use double precision followed by source
rounding to avoid CPU-dependent float32 approximations. This matters for draw
consumption as well as values: a one-unit exponential difference can flip a
rejection test for extremely unbalanced shapes. The logarithm helper is shared
with the gamma sampler.

Legacy shapes must fit float32 and satisfy the source minimum check. The code
rejects values **below** `1e-37`, although its argument comments say strictly
greater. Fortran compares to a float32 constant; C compares its rounded float32
argument to a double constant. Thus the float32 representation of `1e-37` is
accepted by Fortran and rejected by C; the next larger float32 is accepted by
both. Default mode accepts smaller positive shapes.

GENBET uses `expmax=87.49823`, an internal `1e38` sentinel for large weights,
and a `1e-37` guard on a logarithm argument. For shapes at most one it can
combine `v+log(a)` before exponentiation to avoid intermediate overflow.
These are internal source guards, not a final-result truncation warning.
Nonfinite coefficients or zero rejection scale raise `ArithmeticError` rather
than sampling with invalid arithmetic. Nonfinite/out-of-range outputs also
raise an error. Finite precision still limits the accuracy of extreme legacy
parameter combinations; compatibility is with the source arithmetic.

The total draw budget is `max(100000, 8*size)` unless `max_attempts` is supplied.
Budget exhaustion and all other failures leave generator state unchanged.
Other streams and reset anchors are untouched. Parameter failures are reported
before draws; invalid derived legacy coefficients are reported before sampling.

`tools/reference_randlib_beta.py` compiles unchanged C, Fortran 77 and Fortran
95 sources with recorded hashes, compiler versions and flags. Its fixture
contains 231 cases / 4,620 values and states: BB/BC, the shape-one boundary,
swapped and mixed shapes, tiny and large shapes, source minimum boundaries,
two streams, antithetic flags and engineered maximal-uniform cases exercising
overflow guards. Tests require exact native states and values within `6e-6`
relative plus two float32 subnormal units for transcendental rounding. They
also check empirical moments/CDFs, inverse-CDF round trips, scalar/batch
identity, uniform and reflection identities, endpoints, invalid inputs and
rollback. The benchmark includes 10,000 default beta draws with shapes 2 and 3.

Remaining count and multivariate samplers and the final RANDLIB coverage/performance
audit remain pending.

## Binomial sampling

```python
bank = RandlibGenerator()
counts = bank.binomial(1000, n=100, p=0.3)
original = bank.binomial(1000, n=100, p=0.3, legacy=True)
original_c = bank.binomial(1000, n=100, p=0.3, legacy=True, source="c")
```

`binomial(size=1, *, n=1, p=0.5, legacy=False, source="fortran",
max_attempts=None)` returns read-only int64 counts. Trial count `n` must be an
integer in `0..2**53-1`; probability must be finite in `[0,1]`. Mean is `n*p`
and variance is `n*p*(1-p)`. Default mode uses SciPy's vectorized binomial
quantiles and one raw uniform per output. It verifies that each result is an
integer in range and brackets the uniform between `CDF(k-1)` and `CDF(k)`,
allowing numerical tolerance `64*float64_epsilon + 1e-10*min(u,1-u)`.
Invalid quantiles or probability brackets raise `ArithmeticError` without
committing state.

Legacy mode implements IGNBIN's inversion when `n*min(p,1-p) < 30`, and BTPE
otherwise. BTPE uses triangular/parallelogram/exponential regions, explicit
PMF ratios, squeeze bounds and the source's final Stirling expression.
Probability reflection and float32 rounding follow the archived code. C and
Fortran differ in promoted constants, tail-expression evaluation and `q**n`:
C evaluates a double power and rounds; Fortran uses single-precision integer
exponentiation. The shared logarithm helper avoids CPU-specific float32 log
approximations. Constants are local, so changes in trials or probability do
not retain stale setup state.

Zero trials and probabilities 0 or 1 still consume uniforms. Legacy inversion
uses a strict `u < f` acceptance and restarts after its index exceeds 110.
Consequently, when C RANF rounds a raw draw to exactly 1, even a degenerate
request can consume an extra uniform. Empty batches consume nothing.

Legacy `n` is limited to `0..2147483646` so `n+1` fits the common signed-32-bit
source contract. Probabilities that round to an endpoint from an interior
value are rejected. Other source rounding remains visible: for example,
`n=1_000_000_000, p=1e-8` rounds the complement to 1 and yields only zeros in
Fortran legacy mode, despite a mathematical mean of 10. Default mode uses the requested
probability. Legacy compatibility is not a promise of numerical accuracy for
extreme source parameters.

The source Fortran squeeze expression squares an integer distance. Python
raises before that signed-32-bit square would overflow, rather than reproducing
undefined integer overflow. Explicit PMF evaluation is capped at 100,000 steps
per proposal. The total raw-draw budget defaults to `max(100000, 8*size)`;
`max_attempts` may override it. These guards, invalid results and all other
failures leave generator state unchanged. Default mode does not use the legacy
integer-square or explicit-PMF paths.

`tools/reference_randlib_binomial.py` compiles unchanged C, Fortran 77 and
Fortran 95 implementations, with source hashes and compiler/flag provenance.
The fixture contains 387 cases / 7,740 counts and component states: inversion,
the mean-30 switch, BTPE, a targeted final-Stirling case, reflection, mixed
parameter sequences, degenerate laws, large trial counts, small probabilities,
maximal raw uniforms, two streams and antithetic flags. Tests require exact
native counts and states. Additional tests check moments/CDFs, discrete
quantile brackets, scalar/batch identity, endpoint consumption, source overflow
protection and rollback. The benchmark includes 10,000 default binomial draws
with 1,000 trials and probability 0.3.

## Poisson sampling

```python
from mdanderson_stats import RandlibGenerator

bank = RandlibGenerator()
counts = bank.poisson(1000, mu=20)
original = bank.poisson(1000, mu=20, legacy=True)
original_c = bank.poisson(1000, mu=20, legacy=True, source="c")
```

`poisson(size=1, *, mu=1.0, legacy=False, source="fortran",
max_attempts=None)` returns a read-only `int64` array. `mu` is the nonnegative
mean and variance. Default mode uses vectorized SciPy Poisson quantiles and
checks the probability bracket using the smaller tail. Each output consumes
one basic generator draw, including `mu=0`. The finite underlying uniform
grid limits the tails that can be reached. Means must be finite and at most
`2**53 - 1`; a quantile outside that exact-integer range, a nonfinite result,
or an invalid probability bracket raises `ArithmeticError` without advancing
the public generator.

Legacy mode expresses IGNPOI / RANDOM_POISSON's Ahrens–Dieter modified-normal
algorithm. Below a float32 mean of 10 it uses cumulative Poisson probabilities
through count 35, restarting with another uniform if the table is exhausted.
At or above 10 it uses the source normal proposal, immediate and squeeze
acceptance, quotient tests, and exponential hat rejection. It reuses the
validated source normal/exponential primitives and preserves C/Fortran
constant promotion, factorial powers, polynomial coefficients, and uniform
rounding. Source logarithms and exponentials are evaluated in double precision
then rounded to float32 for Fortran mode to avoid known platform-dependent
single-precision vector-library rounding differences.

The original Fortran sources save the table's length and cumulative mass but
omit `SAVE` for `pp(35)`. With automatic local storage, native repeated calls
in the reference driver produced incorrect increasing counts. The recorded
reference builds use gfortran's `-fno-automatic` to retain that table without
editing the archived source. C already declares the table static. Python
constructs the table per request and retains it for the whole batch, making
scalar, batched and alternating-mean calls independent of process-global
cache or stack contents. This deliberately repairs the table lifetime;
compatibility refers to the recorded builds with that storage setting.

Legacy means must round to float32 values in `[0, 2**31)` without positive
underflow to zero. Counts outside the common signed 32-bit native range
raise `ArithmeticError` and roll back the entire request. Float32 rounding
remains visible: at `mu=1e-8`, the initial probability rounds to one and
legacy sampling always returns zero, while default mode retains a nonzero
chance of a positive count. Large legacy means also lose count resolution.
Use default mode for the mathematical distribution instead of reproducing
these single-precision limitations.

`max_attempts` bounds all underlying uniforms, including those used in nested
normal and exponential sampling. It defaults to `max(100000, 8*size)`.
Exceeding it, overflowing the count, or reaching an inherited source-table
guard leaves the selected generator unchanged. Empty batches consume no
randomness; zero-mean batches still consume one uniform per result.

`tools/reference_randlib_poisson.py` compiles the archived C, Fortran 77 and
Fortran 95 implementations. Its fixture records compiler flags, storage
settings and source hashes for 336 cases containing 6,720 counts and component
states. Tests require exact agreement, covering means around the algorithm
boundary, tiny and zero means, streams 1 and 32, antithetic draws, maximum
uniforms and alternating means. Additional tests check moments and CDFs,
scalar/batch agreement, quantile brackets, parameter validation, count-range
failures and rollback. The benchmark includes 10,000 default draws at `mu=20`
compared with repeated calls to the same Python API.

## Negative-binomial sampling

```python
from mdanderson_stats import RandlibGenerator

bank = RandlibGenerator()
failures = bank.negative_binomial(1000, n=10, p=0.3)
original = bank.negative_binomial(1000, n=10, p=0.3, legacy=True)
original_c = bank.negative_binomial(1000, n=10, p=0.3, legacy=True, source="c")
```

`negative_binomial(size=1, *, n=1, p=0.5, legacy=False,
source="fortran", max_attempts=None)` returns read-only `int64` counts of
**failures before `n` successes**, where each trial succeeds with probability
`p`. The mean is `n*(1-p)/p` and variance is `n*(1-p)/p**2`. This resolves the
archived C description's ambiguous reference to the number of trials:
the implemented gamma–Poisson mixture counts failures, not total trials.
`n` must be a positive integer at most `2**53 - 1`; default `p` is in `(0, 1]`.

Default mode uses vectorized SciPy negative-binomial quantiles with checks on
the discrete probability bracket using the smaller tail. It consumes one
basic uniform per output, including the `p=1` extension that always returns
zero. Nonfinite quantiles, results outside `[0, 2**53 - 1]`, or inaccurate
probability brackets raise `ArithmeticError` without advancing the generator.
As with the other default samplers, the uniform generator's finite grid limits
reachable tails.

Legacy mode implements IGNNBN / RANDOM_NEGATIVE_BINOMIAL: draw a standard gamma
with shape `float32(n)`, divide by the float32 rate `p/(1-p)`, then draw a Poisson
count with that random mean. Both original C and Fortran use float32 arithmetic
for the rate. Nested gamma and Poisson algorithms retain their source-specific
rounding and consume a variable number of uniforms from the selected stream.
The implementation reuses their validated primitives with one local state and
one budget for the entire requested batch.

Legacy `n` is at most `2147483647`; legacy `p` must remain strictly between zero
and one after float32 conversion. A nonfinite or out-of-range Poisson mean,
count overflow, or inherited source guard raises `ArithmeticError` and rolls
back the entire request. Source single-precision approximations, loss of
resolution at large parameters, and the repaired Poisson table lifetime
(described above) remain applicable. Fortran reference builds retain local
storage with `-fno-automatic`; the native source itself is unchanged.

`max_attempts` defaults to `max(100000, 8*size)` and limits every underlying
uniform, including rejected gamma/Poisson proposals and their normal and
exponential subdraws. Empty batches consume no randomness. Parameter failures
and unsuccessful batches leave generator state unchanged.

`tools/reference_randlib_negative_binomial.py` records 336 native cases from
C, Fortran 77 and Fortran 95: 6,720 exact counts and component states. Cases
cover gamma coefficient regimes, small/large Poisson means, near-endpoint
probabilities, streams 1 and 32, antithetic and maximum-uniform starts, and
alternating parameters. Additional tests check moments, distribution CDFs,
scalar/batch identity, the independent geometric inverse formula at `n=1`,
zero counts at `p=1`, parameter limits and overflow/budget rollback. The
benchmark measures 10,000 default draws at `n=10, p=0.3` against repeated
scalar calls to the same Python API.

## Multinomial sampling

```python
from mdanderson_stats import RandlibGenerator

bank = RandlibGenerator()
counts = bank.multinomial(1000, n=100, p=[0.2, 0.3, 0.5])
original = bank.multinomial(1000, n=100, p=[0.2, 0.3, 0.5], legacy=True)
original_c = bank.multinomial(1000, n=100, p=[0.2, 0.3, 0.5], legacy=True, source="c")
```

`multinomial(size=1, *, n=1, p=(0.5, 0.5), legacy=False,
source="fortran", max_attempts=None)` returns a read-only `int64` array with
shape `(size, K)`. Each row contains nonnegative category counts summing to
`n`. `p` supplies **all K category probabilities**, each finite and in `[0, 1]`,
with their sum within `1e-12` of one. Default mode treats the accepted vector
as normalized probabilities through ratios to the remaining probability mass;
it does not accept arbitrary unnormalized weights. `n` is a nonnegative integer
at most `2**53 - 1`. Both `K` and `size*K` are bounded by the generator's
`max_draws` limit to constrain memory use.

Default mode samples conditional binomial quantiles, vectorizing across rows.
For each category except the last, it draws from a binomial with the remaining
trial count and probability `p[j]/sum(p[j:])`. Reverse cumulative sums retain
small remaining tails without subtracting them from one. The final category
receives the leftover count. Each conditional quantile is checked for bounds,
integrality and its probability bracket before committing generator state.

The uniform schedule is row-major: `K-1` consecutive uniforms per observation,
even after all trials have already been assigned or when `n=0`. This preserves
scalar/batch equality while allowing vectorization across observations. A single
category is supported in default mode: it receives all `n` events and consumes
no randomness. Zero-probability categories and a zero-probability final category
are also supported. Empty batches consume no randomness.

Legacy mode follows GENMUL / RANDOM_MULTINOMIAL and reuses the validated
source binomial sampler. It requires at least two categories and
`n <= 2147483646`, the common safe range of the underlying IGNBIN arithmetic.
The first `K-1` probabilities are converted to float32 and added sequentially;
their sum must be at most `float32(0.99999)`, matching the source restriction.
The last category is the implicit residual after those source-rounded
probabilities; the full Python input vector must still pass its sum-to-one
check. Positive probabilities that underflow float32 are rejected.

Source conditional probabilities use sequential float32 subtraction from one.
Legacy sampling stops a row as soon as its remaining count reaches zero, so
consumption is variable. With zero trials it still calls the first binomial
sampler before stopping; the source C unit-uniform restart can consume extra
draws even then. Binomial inversion/BTPE rounding limitations and overflow/work
guards remain applicable. A numerical failure or exhausted budget rolls back
the entire batch, including earlier successful rows. `max_attempts` defaults
to `max(100000, 8*size*(K-1))` and counts every underlying uniform.

`tools/reference_randlib_multinomial.py` compiles unchanged C, Fortran 77 and
Fortran 95 sources. Its 288 cases record 5,760 multinomial vectors and component
states, requiring exact agreement. They cover two through ten categories,
zero categories, residual-boundary probabilities, tiny probabilities with
large trial counts, inversion and BTPE, zero trials, stream selection,
antithetic starts and maximum uniforms. Additional tests verify row totals,
means, covariances, binomial marginals, scalar/batch identity, draw schedules,
resource limits and rollback. The benchmark uses 10,000 default observations
with `n=100, p=[0.2, 0.3, 0.5]` against repeated scalar calls to this Python API.

## Multivariate-normal sampling

```python
from mdanderson_stats import RandlibGenerator, RandlibMultivariateNormal

model = RandlibMultivariateNormal([1, -2], [[4, -1], [-1, 2]])
bank = RandlibGenerator()
vectors = bank.multivariate_normal(model, 1000)
more_vectors = bank.multivariate_normal(model, 1000)

original = RandlibMultivariateNormal([1, -2], [[4, -1], [-1, 2]], legacy=True, source="fortran95")
source_vectors = bank.multivariate_normal(original, 1000)
```

`RandlibMultivariateNormal(mean, covariance, *, legacy=False,
source="fortran", max_dimension=256)` prepares the SETGMN parameters once.
It owns immutable snapshots of the mean and lower triangular Cholesky factor;
inputs are not mutated and later input edits cannot affect the model. Multiple
models can be interleaved without replacing global parameters. The configurable
`max_dimension` bounds factorization memory and work; dimensions must be positive.

`bank.multivariate_normal(parameters, size=1, *, max_attempts=None)` returns a
read-only float64 array of shape `(size, dimension)`. The product
`size*dimension` must not exceed the bank's `max_draws`. Default mode requires
a finite, exactly symmetric, positive-definite covariance matrix, computes
its double-precision lower factor `L`, and produces `Z @ L.T + mean` with
vectorized inverse-normal draws. One uniform is consumed per component, in
row-major order. Reusing parameters avoids repeating the factorization.
Scalar and batched results can differ by final floating-point rounding in
matrix multiplication, but consume identical generator states.

Legacy mode preserves SETGMN / GENMN's float32 factorization and transform.
As in the source, only the upper covariance triangle is used; lower entries
are ignored, including nonfinite lower entries. Nonzero mean/upper-covariance
values must fit float32 without underflow. Matrices that lose positive
definiteness after conversion or fail a Cholesky pivot are rejected.
Singular positive-semidefinite covariances are not supported by either mode.

The `source` choices are `"fortran"` (Fortran 77), `"fortran95"`, and `"c"`,
and nondefault source selection requires legacy mode. They retain the recorded
builds' different SDOT accumulation: sequential additions for Fortran 77,
five-term grouped additions for C, and a prefix plus remaining DOT_PRODUCT
for Fortran 95. Source normal tables and arithmetic are reused, followed by
sequential float32 multiplication/addition for each transformed component.
Legacy factors and vectors are matched exactly after float32 conversion
against the reference builds; no claim is made about all compiler optimization
or floating-point contraction settings.

The archived Fortran 95 setter writes to the module's public allocatable
`param` without allocating it, causing the initial reference call to crash.
The reference driver allocates that workspace before calling the unchanged
setter. Python manages storage inside the prepared object and needs no manual
workspace or initialization. The fixture records this driver requirement.

An empty batch consumes no randomness. `max_attempts` defaults to
`max(100000, 4*size*dimension)` and includes all rejected source normal draws.
Budget exhaustion or nonfinite sampled output leaves the bank unchanged.
Invalid parameters are rejected when constructing the prepared object, before
sampling can change generator state.

`tools/reference_randlib_multivariate_normal.py` records 240 native cases:
4,800 vectors and component states plus their packed means/Cholesky factors.
They include dimensions 1, 2, 3, 7, 8 and 12, positive and negative correlations,
near-singular covariance, small/large scales, antithetic samples, streams 1 and
32, and maximum uniforms. Tests compare native factors, vectors and states;
also verify empirical means/covariances, reconstructed covariance, independent
immutable models, normal identity, scalar/batch behavior, resource limits,
parameter errors and rollback. The benchmark prepares one three-dimensional
model per run and compares 10,000 batched observations with repeated calls
to the same Python API, reusing the model in both cases.

RANDLIB's final archive coverage/performance audit remains pending.

## Phrase and clock seeding

```python
from datetime import datetime
from mdanderson_stats import RandlibGenerator

bank = RandlibGenerator()
base_seed = bank.set_phrase("study A", stream=1)
clock_seed = bank.set_time(datetime(2026, 9, 8, 12, 34, 56, 789000))
replay = RandlibGenerator(clock_seed, stream=bank.stream)
```

`set_phrase(phrase, *, stream=1, source="fortran")` hashes the full ASCII
phrase, resets all 32 initial/block/current states, selects the requested
stream and returns the base seed pair. Antithetic flags are retained.
Trailing spaces are ignored; an empty/all-space phrase uses the source default
pair. The noninteractive API accepts full phrases rather than truncating to
the legacy interactive dialog's 80-character buffer. The Fortran hash reuses
`ranlist_seeds`; unsupported ASCII characters use its original fallback code.

C's PHRTSD has a different table, including an extra backslash before the
quotation mark. This changes punctuation hashes, including time strings with
a decimal point. `source="c"` preserves the C mapping and its special handling
of the final table character `/`. The source reads beyond its lookup table
for unknown characters. Python rejects those C-mode inputs instead of
reproducing undefined memory access; this includes internal spaces, tabs,
NUL and `~`. Trailing spaces are still removed first. Use the default Fortran
mapping for arbitrary ASCII text.

`set_time(moment=None)` hashes `HHMMSS.mmm`, matching the Fortran time-of-day
seeding helper. With no argument it reads the local clock once. An explicit
`datetime` makes this deterministic: its displayed hour/minute/second fields
are used as supplied, microseconds are truncated to milliseconds, and neither
the date nor timezone offset enters the hash. The method retains the selected
stream and antithetic flags, resets all streams, and returns the base seed
pair for recording/replay. The same time of day produces the same seeds;
this is the archived reproducibility convention, not an entropy guarantee.

The Fortran 95 `user_set_generator` module marks its documented convenience
routines private. Its interactive phrase routine computes the hash but omits
the call that applies it to the generator. Python provides explicit methods
and applies the reset correctly. The reference driver exposes only the private
hash routine for measurement, keeping its body unchanged; fixture provenance
records the modification and original source hashes.

`tools/reference_randlib_seeding.py` validates 180 phrase/stream cases across
C, Fortran 77 and Fortran 95: base seeds plus 1,800 raw values and component
states. Cases include punctuation, clock strings, long phrases, trailing
spaces, streams 1 and 32 and antithetic flags. Undefined C lookup inputs are
excluded from native execution and tested as Python errors. Additional tests
verify all-stream/block reset, stream retention, actual-clock replay,
millisecond formatting, immutable failure state and shared Fortran hashing.
