# RANDLIB

Catalog entry 27 is partial. The 32-stream generator bank and state controls are
implemented, along with bounded uniforms, permutations, exponential, normal, gamma,
central/noncentral chi-square, F and beta sampling. Count and multivariate samplers and the final
archive coverage/performance audit remain pending.

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
[randlib-benchmark.json](randlib-benchmark.json). Count and multivariate distributions and RANDLIB's final coverage audit
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

Count and multivariate samplers and the final RANDLIB coverage/performance
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
selects the C expression promotions and squared-uniform calculation. Shared
logarithm evaluation uses double precision followed by source rounding, as in
the gamma sampler, to avoid CPU-dependent float32 logarithm approximations.

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

Count and multivariate samplers and the final RANDLIB coverage/performance
audit remain pending.
