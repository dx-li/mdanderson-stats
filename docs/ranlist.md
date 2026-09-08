# RANLIST randomization lists

Catalog entry 29 is partial. The phrase-to-seed conversion and indexed random
streams and unrestricted treatment allocation are implemented. Restricted
allocation, fixed/random balance points, strata/list management, interactive
assignment and reports remain pending.

The [official entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/29)
lists version 1, modified August 23, 2002. The archive filename contains two
spaces: `RANLIST  _V1.tar.gz`. It contains `source/ranlist.f`, a plain-text
`source/ranlist.doc`, `source/readme`, and `win32/ranlist_1.2_se.exe`. The source
and manual identify version 1.1, July 1992; the executable name identifies 1.2.
This implementation currently validates against the archived Fortran source,
not the Windows executable. Original source, manual and executable are not
bundled. The readme and manual's legal section are retained under `notices`.

## Reproducible indexed streams

```python
from mdanderson_stats import ranlist_integers, ranlist_seeds, ranlist_uniform

seed = ranlist_seeds("trial 123")
raw = ranlist_integers([1, 2, 1000], seed=seed, stream=2)
u = ranlist_uniform([1, 2, 1000], seed=seed, stream=2)
source_u = ranlist_uniform([1, 2, 1000], seed=seed, stream=2, legacy=True)
```

`ranlist_integers` reproduces IGNLGI, the two-component modular generator used
by RANLIST. It returns integers from one through 2,147,483,562. Default seeds
are (1234567890, 123456789). Explicit seeds must be integers in
1..2147483562 and 1..2147483398 respectively.

Positions are **one-based** draw numbers, and stream numbers are **one-based**
from one through 32, matching RANLIST's stratum/generator numbering. Positions
retain their input shape, may repeat or be unsorted, and must be positive
integers below 2^53. Returned arrays are read-only. Empty position arrays are
valid. Calls have no shared mutable state, so requesting a later position does
not change an earlier result or another stream.

SETALL separates stream starts by 2^50 draws. Requests beyond that spacing
continue mathematically into subsequent stream positions; choosing different
stream numbers does not prevent overlap if those ranges are exceeded. INITGN's
2^30-draw block jumps can be represented by adding multiples of 2^30 to a
position. The implementation uses vectorized modular exponentiation to reach
requested positions directly, with logarithmic work in the largest position.
All modular products fit in signed 64-bit integers; this replaces MLTMOD's
31-bit overflow-avoidance arithmetic without changing the integer sequence.

`ranlist_uniform` divides by 2147483563 in double precision by default, yielding
values strictly inside (0,1). `legacy=True` reproduces RANF: it rounds the raw
integer to float32, multiplies by the float32 representation of 4.656613057e-10,
and returns that result as float64. The rounded constant equals 2^-31, so the
legacy and default scalings differ. Legacy conversion remains strictly below
one, with maximum 1-2^-24. This option reproduces the source uniform values,
not NumPy Generator sequences.

`ranlist_seeds` implements PHRTSD. Trailing ASCII spaces are removed; tabs and
other characters are not stripped. Case matters. Characters absent from the
source's lookup table use its fallback code, so distinct phrases can map to
the same seeds. An empty/all-space phrase retains the original default seeds
rather than reducing them modulo 2^30. Non-ASCII text is rejected because its
encoding into the original single-byte interface is ambiguous. Returned values
follow the source formula, including its modulo arithmetic; seed validation is
performed when the resulting pair is used to generate draws.

These functions provide the numerical foundation for reproducing RANLIST.
Bounded-integer rejection sampling, treatment permutations and restricted
allocation remain separate pending components.

## Validation

`tools/reference_ranlist_random.py` extracts the required unchanged program
units from the archive and compiles local Fortran drivers. Fixtures record the
source hash, compiler and flags. Thirty-six cases cover three seed pairs,
streams 1/2/17/32, and blocks 0/1/3, with six draw positions per case. Integer
draws match exactly; legacy uniforms match the native float32 values. Nine
phrase cases cover blank input, case, punctuation, embedded/trailing spaces
and unknown characters.

Further tests compare indexed results with an independent scalar recurrence,
check stream-spacing and very large modular powers, preserve array shape and
ordering, and validate inputs and float scaling. These comparisons validate
only the named random-stream/seed routines, not the pending RANLIST workflows.

## Unrestricted treatment allocation

```python
from mdanderson_stats import ranlist_seeds, ranlist_unrestricted

fit = ranlist_unrestricted([1, 2, 1000], [1, 2, 3], seed=ranlist_seeds("trial 123"))
print(fit.treatments)  # One-based treatment numbers, matching the patient shape
source = ranlist_unrestricted([1, 2, 1000], [1, 2, 3], legacy=True)
```

`ranlist_unrestricted` implements IGTUT and the weight normalization used by
GENLST and WRKLST. Supply one through 20 strictly positive relative treatment
weights; their sum need not be one. Patient numbers and stream numbers follow
the indexed generator's one-based convention. Each patient number refers to
that position **within the selected stream**, not an overall cross-stratum
arrival number. The kernel accepts all 32 generator streams; the archived
interactive list interface limits its named strata to 20.

The result retains copied patient numbers and weights, normalized probabilities,
cumulative boundaries, uniforms, treatment assignments, seed, stream and legacy
setting. Arrays are read-only. Requested patients may be unsorted, repeated or
in any array shape. There is no shared patient counter: querying other patients
or streams cannot change an assignment. This provides indexed assignments;
interactive enrollment counters and saved lists are not yet implemented.

Default normalization scales weights by their maximum before summing, preventing
overflow for large finite weights. Cumulative probabilities are double precision,
and the last boundary is set to one. Positive weights whose intervals cannot
be represented distinctly in double precision are rejected. Assignment uses
the first cumulative probability greater than or equal to the uniform draw,
matching IGTUT's inclusive boundary. An exactly equal draw belongs to the lower
numbered treatment.

`legacy=True` first converts weights to float32, sums them sequentially, divides
each by that total, and sequentially accumulates the cumulative probabilities.
It also uses the original float32 uniforms. These details can change assignments
near boundaries. Legacy preserves collapsed intervals from source rounding;
weights and their total must still be representable as positive finite float32
values. It does not silently force the final cumulative value to one. If a
requested uniform exceeds that value, Python raises `ArithmeticError`: the
source would continue beyond its valid treatment probabilities. For example,
twelve equal float32 probabilities sum to 0.9999998807907104, below the largest
RANF draw. Default normalization handles that case without the source defect.

This numerical API accepts relative weights directly. It does not impose the
archived parameter file's F6.3 text rounding; legacy file parsing is separate
future work. Extreme probabilities are also subject to the finite resolution
of the underlying random-number stream.

`tools/reference_ranlist_unrestricted.py` compiles unchanged IGTUT and random
routines with the source's weight-accumulation steps in a local driver. Forty
cases cover five weight vectors, two seed pairs and four streams, including an
engineered draw exactly on a cumulative boundary. All 520 retained assignments
match exactly, as do the float32 cumulative probabilities. Independent tests
use a scalar integer recurrence and rational weighted boundaries, check fixed
large-sample allocation frequencies, weight scaling, indexed reproducibility,
input snapshots and unsupported numerical cases.
