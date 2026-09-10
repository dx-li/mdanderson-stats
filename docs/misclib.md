# Misclib: shared numerical library and scalar maximization

Misclib is a library of statistical-software support routines by the MD Anderson
Section of Computer Science, Department of Biomathematics. This port reuses the
existing CDFLIB implementations where the archived procedure bodies agree.
Catalog entry 87 remains **partial**: its numerical methods are available, while
some formatting and file/prompt utilities still need ports
or a source-contract audit.

## Bounded scalar maximization

```python
from math import log, log1p
from mdanderson_stats import fun_max, set_fun_max

result = fun_max(
    lambda p: 3 * log(p) + 7 * log1p(-p),
    local=set_fun_max(0.01, 0.99, abs_tol=1e-10, rel_tol=1e-10),
)
print(result.x, result.value)  # approximately 0.3, -6.108643...
```

`set_fun_max(low_limit=-1e35, hi_limit=1e35, *, abs_tol=1e-5,
rel_tol=1e-5, max_evaluations=1000)` returns a fresh `FunctionMaximizer`.
Choose bounds meaningful for the objective. Bounds and their difference must
be finite, with low < high. Tolerances must be nonnegative, at least one positive,
and relative tolerance < 1. The objective must return a finite scalar.

The Brent parabolic/golden-section search starts at the midpoint, as the source
does. Its stopping tolerance is
`max(sqrt(machine_epsilon), rel_tol)*abs(x) + abs_tol/3`, with a one-ULP floor
for representability. The source prose's simpler `max(abs_tol, rel_tol*x)`
is not its implemented stopping rule. The returned `FunctionMaximum` contains
`status`, `x`, `value`, the retained `lower`/`upper` interval and `evaluations`.
It approximates a maximum for a unimodal function. It does not certify a global
maximum of a multimodal objective. A boundary maximum is approached from within
the interval; endpoint evaluation is not automatically added.

For reverse communication:

```python
from mdanderson_stats import rc_fun_max, set_fun_max

state = set_fun_max(-2, 2)
request = rc_fun_max(state)
while request.status == 1:
    request = rc_fun_max(state, -((request.x - 0.4) ** 2))
print(request.x)
```

Each status-1 result requests `f(x)`, not its negative. The first call takes no
function value. Completion has status 0; construct a fresh state before another
search. `state.result` is an immutable snapshot. Independent states can be
interleaved or nested; there is no default shared Fortran SAVE storage. Budget
exhaustion raises `ArithmeticError` and retains a status -2 result with the best
value seen. An invalid submitted value raises without consuming the request.

The implementation retains the source's signed previous-step criterion for
accepting a parabola, which conservatively rejects some leftward interpolations.
Nonfinite interpolation intermediates fall back to golden-section refinement.
Midpoints avoid sum overflow, and rounded steps stay within the bounds. These
changes address numerical limits without changing the ordinary search method.

## Shared source audit and remaining coverage

A procedure-by-procedure comparison against the pinned CDFLIB90 archive removes
comments, whitespace, redundant IMPLICIT NONE declarations, terminal RETURNs and
the `biomath_` module prefix. It preserves executable expressions. All 35 math,
five string and eight root-finder procedure blocks compare identically; hashes
and individual results are in `misclib-shared-source.json`. This is source
cross-reference evidence, not a claim that Python reproduces every native
intermediate or legacy failure mode. Existing CDFLIB notes document numerical
repairs, domains and Python semantics.

| Misclib source | Python coverage / next work |
|---|---|
| `mathlib_mod` | All 35 existing public functions retain their source names: `algdiv`, `alngam`, `alnrel`, `apser`, `basym`, `bcorr`, `betaln`, `bfrac`, `bgrat`, `bpser`, `bratio`, `brcmp1`, `brcomp`, `bup`, `erf`, `erfc1`, `esum`, `evaluate_polynomial`, `exparg`, `fpser`, `gam1`, `gamln`, `gamln1`, `gamma`, `grat1`, `gratio`, `gsumln`, `log_beta`, `log_bicoef`, `log_gamma`, `psi`, `rcomp`, `rexp`, `rlog`, `rlog1`; see [CDFLIB coverage](cdflib90-coverage.md) |
| `zero_finder_mod` | Existing `set_zero_finder`, `interval_zf`, `step_zf`, `rc_interval_zf`, `rc_step_zf`, `final_zf_state`; local state replaces source globals. Python uses safeguarded interpolation/bisection rather than the source TOMS 748 kernel; see [root methods](cdflib-root.md) |
| `strings_mod` | Existing ASCII case conversions and reentrant `qlex`; [string semantics and native validation](cdflib-strings.md) |
| `max_fun_mod` | New `set_fun_max`, `fun_max`, `rc_fun_max` and per-search state, described above |
| `constants_mod` | Existing `cdflib_constants` appears equivalent; final source-contract audit pending |
| `sort_mod`, `sort_permutation_mod` | Matrix-column sorting, gather indices, direct/reversed gathers and callback contracts implemented below. Existing `sort_list` offers list sorting; its Misclib-specific source comparison remains pending |
| `format_number_mod`, `print_it_mod`, `format_specs` | Number/template formatting needs a Misclib-specific audit/port |
| `get_values_from_user_mod`, `open_file` | Existing console helpers offer related behavior; exact prompting/file contracts not yet audited |
| `interface_mod`, build/install files | Fortran interfaces and installation need final reconciliation with Python packaging |

The manual mistakenly prints `2/sqrt(2*pi)` in its erf/erfc definitions. The
executable computes the standard `2/sqrt(pi)` normalization, as the existing
Python implementation does. The prose also understates the mathematical scope:
the downloaded source includes incomplete beta/gamma and additional support
routines, all included in the 35-procedure crosswalk above.

## Focused validation and provenance

The unchanged `max_fun_mod.f90` was compiled with gfortran `-O2 -fcheck=all`.
Six cases cover positive/negative quadratic peaks, a monotone boundary maximum,
a flat quartic peak, a binomial log likelihood and a sine maximum. Python agrees
with native maximizers within 2e-8 and independently known locations. Five cases
also happened to match native positions and evaluation counts exactly; small
log/log1p rounding differences change the likelihood search path, so bitwise
path agreement is not an API guarantee.

`tests/test_misclib_maximum.py` adds interleaved reverse communication, budget
failure, recovery from a rejected nonfinite value, subnormal interval handling,
and finite objectives of magnitude 1e300 over coordinate scales 1e-200 to 1e300.
All requested evaluation points stay within their configured bounds. Native
results are in `tests/fixtures/misclib-maximum-native.json`.

Sources: [catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/87),
[downloaded archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/misclib/misclib_V1.0f.zip).
The archive URL requires the lowercase `misclib` directory. Source hashes are in
`misclib-sources.json`. See the [original legal notice](../notices/mdanderson-misclib-Legal.txt)
for public-domain contributions and separate third-party terms. Original source
files and archives are not redistributed.

## Matrix sorting and permutations

`permutation_sort_matrix(values, *, ncol=None, irow=0, a_gt_b=None)` returns
immutable **zero-based gather indices** for sorting the first `ncol` columns.
The default compares row `irow`. For example:

```python
from mdanderson_stats import permutation_sort_matrix, permute_matrix, sort_matrix

matrix = [[30, 10, 40, 20], [3, 1, 4, 2]]
index = permutation_sort_matrix(matrix)  # [1, 3, 0, 2]
ascending = permute_matrix(matrix, index)
descending = permute_matrix(matrix, index, opt=-1)
assert (ascending == sort_matrix(matrix)).all()
```

The source's introductory rank-vector prose contradicts its executable:
`index[j]` identifies the original column to place at output column `j`.
Further, `opt < 0` means **reverse the gather sequence**, not invert the
permutation mathematically. `opt > 0` gathers directly; `opt == 0` copies the
input and ignores `index`. Python preserves this executable behavior.

`permute_matrix(values, index=None, *, opt=1, ncol=None, nrowus=None)` and
`sort_matrix(values, *, ncol=None, nrowus=None, irow=0, a_gt_b=None)` operate
on the leading `nrowus` rows and `ncol` columns; defaults select the entire
matrix. Other elements are copied unchanged. This makes the source's untouched
or uninitialized destination elements deterministic. Sorting uses column
records, not rows. Numeric dtype is retained, including integers beyond 2**53.
Input must be a 2-D real numeric or Unicode-string array, with at least one row
and at most two million elements; nonfinite numbers and mixed string/number
sequences are rejected. Inputs are never mutated; outputs have independent,
read-only storage. An empty column dimension is allowed.

A custom `a_gt_b(x, y, irow)` receives immutable full original columns and the
integer `irow` unchanged; it returns a scalar boolean indicating whether `x`
belongs after `y`. It must define a consistent ordering. Inclusive comparisons
are accepted, with symmetric results treated as ties. Without a callback,
`irow` is a zero-based row index, and for `sort_matrix` it must lie within the
rows being moved. When moving only a row prefix, custom keys should likewise
use that prefix for parity with the original in-place algorithm. Python computes
all decisions from original records; it does not reproduce a comparator whose
keys change because some fields stay in place during a native swap.

Ties retain input order, improving on the original unspecified quicksort tie
ordering. Strings are compared with blank padding to the array's character
width, including for callback arguments; stored output strings retain their
input contents. Numeric default sorting uses NumPy's compiled stable sort and
array indexing. Custom comparators use Python's comparison sorter.

Focused reference checks compile the unchanged `sort_mod.f90` and
`sort_permutation_mod.f90` with `-O2 -fcheck=all`. Double, single and integer
variants agree on single-row and custom multirow keys, gather indices, partial
row/column movement and all three option signs. A separate character case checks
blank padding against punctuation and a tab. Results are in
`tests/fixtures/misclib-sort-native.json`; tests also check stable ties, exact
large integers, malformed permutations and immutable ownership.
