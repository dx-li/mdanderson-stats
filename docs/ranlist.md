# RANLIST randomization lists

Catalog entry 29 is partial. The phrase-to-seed conversion and indexed random
streams, unrestricted allocation and restricted allocation with fixed/random
balance points, named list specifications and per-stratum enrollment/inquiry
file persistence and printable reports are implemented. The final coverage
and performance audit remains pending.

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

These functions also supply the integer streams used by the treatment
allocation APIs below.

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
enrollment counters are provided by `RanlistSession` below; saved parameter
files are supported by the persistence APIs below.

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


## Restricted treatment allocation

```python
from mdanderson_stats import ranlist_restricted

fixed = ranlist_restricted([1, 2, 3, 4, 5, 6], [1, 2], balance=(2, 2))
random = ranlist_restricted([1, 20, 100], [1, 2], balance=(1, 4))
source = ranlist_restricted([1, 20, 100], [1, 2], balance=(1, 4), legacy=True)
```

`counts` contains one through 20 positive integers. The natural block lists
one-based treatment numbers in order, repeated by their counts: `[1, 2]`
produces `[1, 2, 2]`. Each balance block repeats that natural block K times
before a forward Fisher–Yates permutation. Every completed balance block thus
has exactly K times the requested treatment counts. A partial block need not
have the target proportions. Inclusive ordered `balance=(minimum, maximum)`
bounds control K; equal bounds fix it. The largest possible balance block must
contain at most 500 assignments, matching the archived buffer limit.

Default mode draws a new K on every refill, as described by the manual, and
uses unbiased rejection sampling for each permutation swap. Legacy mode
reproduces three details of IGTRT, GENPRM and IGNUIN:

- IGTRT resets the stream and chooses K once at the start of every patient
  query, so all blocks in a stratum have the same K. This differs from the
  manual's random-refill description.
- IGNUIN accepts its upper rejection boundary inclusively. Its accepted
  integer range contains one extra residue zero, producing a small bias
  toward the lower bound. Default sampling uses an exclusive boundary over
  an exact multiple of the requested range width.
- IGTRT skips exactly block-length-minus-one raw draws per preceding block.
  A rejected permutation draw would consume additional randomness, so this
  indexed source behavior can reuse draws across adjacent blocks. Default
  mode advances through actual permutations, including rejected draws.

The result includes read-only patient numbers, treatment counts, assignments,
one-based inclusive `block_start`/`block_end`, and K in `multiplier` for each
requested patient, plus seed, stream, balance bounds and legacy mode. Scalar,
empty, repeated, unsorted and multidimensional requests are supported. Calls
are independent and deterministic. Stream numbers are 1..32; patient numbers
are positive integers below 2^53.

Legacy mode jumps directly to each distinct requested block with modular
exponentiation and permutes it once, avoiding patient-by-patient replay.
Default mode generates all preceding blocks to locate random balance points
and preserve rejection consumption. `max_blocks=10_000` caps generated blocks,
including preceding blocks in default mode; requests requiring more raise
`ValueError`. Increase the limit deliberately for larger lists. This is a
computation limit, not a truncation of the returned assignments.

`tools/reference_ranlist_restricted.py` compiles unchanged IGTRT, GENPRM,
IGNUIN and their RNG dependencies. Forty-eight native cases cover fixed and
random bounds, one and 20 treatments, unequal counts, four streams, and a seed
that forces the source's rejection boundary. All retained assignments match
exactly. Independent tests check complete-block balance, the manual's refill
rule, forward shuffling, source draw reuse after rejection, distant indexed
queries, shape/order preservation and computation limits. This validates the
restricted numerical kernel; the remaining list workflows are still pending.


## Lists and per-stratum enrollment

```python
from mdanderson_stats import RanlistSession, RanlistSpecification, ranlist_seeds

specification = RanlistSpecification(
    weights=(1, 2),
    restricted=True,
    balance=(1, 4),
    strata=("North", "South"),
    treatments=("Control", "Experimental"),
    title=("Example trial",),
    phrase="trial 123",
    seed=ranlist_seeds("trial 123"),
)
state = RanlistSession(specification)
state, assignments = state.enroll([2, 1, 2])
# Patients 1 in South, 1 in North, and 2 in South.
assert state.current_patients == (1, 2)
previous = state.inquire([1, 2], strata=2)
future = specification.allocate([10, 11], strata=1)
```

`RanlistSpecification` retains an immutable, validated list definition.
`weights` supplies integer counts for restricted lists and relative frequencies
for unrestricted lists. The numerical kernels' defaults and legacy behavior
remain unchanged. Unrestricted lists use `balance=(1, 1)` because they have no
balance blocks. Each of the 1..20 strata maps to its corresponding RNG stream.
Queries use one-based numeric stratum indices; names are descriptive labels.

Labels follow the source's field widths: 1..9 nonblank title lines of at most
80 characters, stratum/treatment names of at most 30, and a phrase of at most
31. They must be printable ASCII; trailing spaces are removed. Names must all
be present or all blank within each category, avoiding the source file's blank
first-name sentinel ambiguity. Empty treatment names default to an unnamed
entry per weight. The seed pair is authoritative; a phrase is metadata and is
not silently rehashed. Supply `ranlist_seeds(phrase)` when creating a list from
a phrase. Persistence APIs below serialize these specifications and their session counters.

`RanlistSession` holds one nonnegative enrollment counter per stratum, initially
zero. `enroll(strata)` processes arrivals in array C order, assigns successive
patient numbers within each stratum, and returns `(updated_session, assignments)`.
Retain the updated session for subsequent enrollment. The original session is
unchanged, including when any part of a batch fails. Treatment evaluation is
batched once per participating stratum rather than repeated for each arrival.
Counters may be supplied explicitly to resume known state, subject to the
numerical patient limit below 2^53. This immutable API provides in-process
state transitions; JSON snapshots provide durable file storage, while concurrent
enrollment coordination is not implemented.

`inquire(patients, strata=...)` accepts only already-enrolled patients and does
not advance counters, matching WRKLST's inquiry rule. The specification's
`allocate` method may also query future positions for list generation. Patient
and stratum inputs broadcast together, preserving scalar/array shape, repeats
and ordering. Results contain read-only `patients`, `strata` and `treatments`.
Empty enrollment batches preserve the counters. Invalid strata, unsupported
patient numbers and resource-limit failures raise before returning new state.

Session tests exercise interleaved and sequential enrollment, resumed counters,
queries, immutable input snapshots, failures without counter advancement,
broadcasting, boundary counts and metadata validation. Restricted/unrestricted
assignments in both modes are compared with the independently native-validated
kernels. The persistence validation below additionally executes the complete archived
program for four enrollment, inquiry and file-update workflows.


## Save, resume and exchange parameter files

```python
from pathlib import Path
from mdanderson_stats import (
    load_ranlist_session,
    ranlist_parameter_text,
    read_ranlist_parameters,
    save_ranlist_session,
)

save_ranlist_session(state, "trial.json")
resumed = load_ranlist_session("trial.json")

# Original files always use the archived allocation behavior.
archived = read_ranlist_parameters(Path("original.par").read_text(encoding="ascii"))
Path("export.par").write_text(ranlist_parameter_text(archived), encoding="ascii")
```

JSON snapshots use the versioned `mdanderson-stats/ranlist` format and retain
full numerical precision, every specification field, algorithm mode, computation
limit and per-stratum counters. Loading validates the complete schema and
numerical contract. Missing/extra fields, duplicate keys, unknown versions,
nonfinite weights and invalid counters are rejected. Saving serializes first,
writes and flushes a temporary file in the destination directory, then atomically
replaces the destination. A failed replacement removes the temporary file and
leaves the existing destination intact. The parent directory must exist. These
files do not coordinate concurrent writers; callers must serialize enrollment
and storage when sharing a list.

`read_ranlist_parameters(text, max_blocks=10_000)` imports the original ASCII
records: header, title count and lines, phrase/seed/settings, optional names,
treatment counts or weights, balance bounds, and patient counters. It retains
the explicit seed pair rather than recomputing it from the descriptive phrase.
The imported session uses `legacy=True`. Blank first-name records denote unnamed
categories. CRLF and omitted trailing padding are accepted; malformed fields,
truncation, overflow asterisks and extra nonblank records are rejected. The parser
supports the canonical decimal F6.3 records written by RANLIST, including their
implied three decimal places when the decimal point is absent. Weights are
converted to float32, matching the source READ. Unrestricted balance fields
are parsed and normalized to `(1, 1)` because MKLST did not initialize them and
they do not affect unrestricted assignment. Non-numeric garbage in those fields
is rejected. Computation limits are not stored in the original format and must
be supplied when importing if a larger value is needed.

`ranlist_parameter_text(session, allow_rounding=False)` exports source-compatible
sessions. Modern sessions must use JSON because the original records cannot
identify their algorithm. The source's I1 counts must be at most nine; I6 counters
must be at most 999999. F6.3 weights must fit six characters and remain positive
after rounding. By default, export rejects any decimal rounding that changes an
effective float32 weight. `allow_rounding=True` explicitly permits that change;
it may change subsequent treatment assignments after re-import. It does not
permit zero weights, overflow or unrepresentable counts. Already representable
source weights, such as 0.1, can be exported without opting into further rounding.
Use JSON to preserve precision and all Python options.

`tools/reference_ranlist_parameters.py` generates two native Fortran fixed-width
record fixtures, including decimal midpoint rounding, then compiles and executes
the entire unchanged archived program. Four workflows cover named/unnamed strata
and restricted/unrestricted allocation. Each opens a Python-exported file with
existing counters, enrolls five interleaved arrivals, inquires about an enrolled
patient, and exits through WRKLST's save path. Assignments, the inquiry result,
updated counters and rewritten file bytes all match Python exactly. Tests also
cover full-precision JSON save/resume, malformed input, explicit rounding and
real filesystem replacement failure. The Windows 1.2 executable remains untested.


## Printable reports

```python
from pathlib import Path
from mdanderson_stats import ranlist_report, ranlist_summary

print(ranlist_summary(state))
Path("enrolled.txt").write_text(ranlist_report(state), encoding="ascii")
Path("planned.txt").write_text(ranlist_report(state, 100), encoding="ascii")
```

`ranlist_summary` lists titles, authoritative seeds, descriptive phrase, algorithm,
restricted balance settings, treatment counts/weights, stratum names and enrolled
counts. It does not generate assignments. Unrestricted summaries omit balance
settings because those settings do not affect their allocations. Both seed and
phrase are shown when available, avoiding ambiguity when the metadata and seed
were supplied independently.

`ranlist_report(session, patients=None, max_rows=100_000)` adds treatment lists.
The default lists the enrolled patients in each stratum. A scalar requests that
many patients in every stratum, including future assignments; a vector specifies
a separate count for each stratum. Zero produces no assignment pages for that
stratum. Printing never changes enrollment counters. Requested counts are shown
separately from enrolled counts in the summary.

Form-feed characters separate pages. Each stratum starts at page one, with
patient number, treatment number/name and a dotted space for patient information.
Page capacity is `27 - number_of_title_lines`, matching the row boundaries of
GENLST's 60-line pagination. The Python report uses simpler header spacing and
retains complete titles/names and full numerical values rather than duplicating
the original's truncation and frequency rounding. It reproduces assignments and
page boundaries, not byte-for-byte typography. The total row limit is validated
before allocation; the specification's `max_blocks` limit also applies. Format
the report before opening its destination for writing, as in the examples, so
validation failures do not replace an existing report.

`tools/reference_ranlist_reports.py` executes the complete archived program for
eight print workflows: restricted/unrestricted allocation, named/unnamed strata
and treatments, and enrolled/requested patient counts. One-line and nine-line
titles exercise different page capacities, with requested lists spanning multiple
pages and an enrolled stratum containing zero patients. Every printed assignment
and page boundary matches Python. Additional tests cover modern random refills,
separate counts per stratum, complete labels, empty lists and resource limits.
