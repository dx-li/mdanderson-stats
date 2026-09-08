# RANDLIB

Catalog entry 27 is partial. The 32-stream generator bank and state controls are
implemented, along with bounded uniforms, permutations and exponential sampling.
Normal, gamma, beta, chi-square, F, count and multivariate samplers and the final
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
