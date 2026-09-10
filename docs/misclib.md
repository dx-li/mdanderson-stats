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
| `format_number_mod` | Integer/single/double number formatting implemented below, including alignment, scaling, trimming and fit reporting |
| `print_it_mod`, `format_specs` | Template-page compilation, fixed-width substitutions and message printing implemented below; screen clearing, pauses and related console utilities remain pending |
| `get_values_from_user_mod` | Existing console helpers offer related behavior; final source-contract audit pending |
| `open_file` | Interactive file selection implemented below with explicit statuses, read/create/append/overwrite and confirmation before mutation |
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

## Number formatting

`format_number(x, *, justi=1, width=20, maxf=1e6, minf=1e-4, ndecf=4,
npe=1, ndece=4, qpad=True)` returns an immutable `FormattedNumber` containing
`text`, `width` and `fit`. It covers the integer, single-real and double-real
procedures of `format_number_mod` with a scalar Python API.

```python
from mdanderson_stats import format_number

assert format_number(12.5, justi=-1, qpad=False).text == "12.5"
assert format_number(1.25e-20, justi=-1).text == "1.2500D-20"
assert format_number(123, width=2).fit is False
```

`justi=-1` returns unpadded text and its used width; `0` centers, placing an odd
extra blank on the right; `1` right-aligns within the requested width. A failed
fit returns `text=None`, `fit=False`, and the unchanged requested width, avoiding
the source's undefined character output. Width can be 0..10000.

Integer inputs are formatted exactly, without conversion to float, and ignore
the floating-format options. Python/NumPy floating values must be finite.
NumPy float32 selects the source single-real E notation and single-precision
thresholds; other supported real inputs use double-real D notation. Thresholds
satisfy `0 <= minf < maxf`. Zero always uses fixed notation. Other values use
fixed notation only when `minf < abs(x) < maxf`: the executable uses strict
inequalities at **both** thresholds, despite inconsistent manual wording.

`ndecf` and `ndece` specify 0..100 decimal places in the displayed fixed number
or scaled exponential mantissa. `npe` is the Fortran scale factor: 1 yields
`1.2500D-20`, 0 yields `0.1250D-19`, and 2 yields `12.5000D-21` for the example
above. Supported `npe` is `1-ndece` through 100, satisfying the native negative
scale restriction. `qpad=False` removes trailing fractional zeros while keeping
at least one fractional digit when precision is positive. With zero decimal
places, the decimal point remains. Like the executable, fit is checked **before**
trailing-zero removal. Defaults are Python conveniences; the source requires
all formatting parameters explicitly.

Decimal arithmetic formats the exact represented binary floating value with
round-to-nearest, ties-to-even. It avoids fixed-size native scratch buffers and
integer conversion when measuring a floating number's width. Documented repairs:

- Formatting 9.999 to two fixed decimal places produces `10.00`; the original
  compiled routine produces `****` and nevertheless reports success because it
  allocates the internal field before rounding. Python measures the rounded
  text and reports failure if the requested field cannot hold it.
- Large integer values retain exact digits, including values beyond 2**53 and
  the signed-int32 minimum. Native integer negation and REAL/log10 digit counting
  can overflow or miscount; Python uses integer string conversion.
- Three-digit exponents retain the E/D letter. For 1e300 the compiled original
  emits `1.0000+300`; Python emits the unambiguous `1.0000D+300`.
- Negative floating zero is displayed as positive zero, matching the ordinary
  source zero branch.

An unchanged `format_number_mod.f90` compiled with `-O2 -fcheck=all` matched 240
ordinary cases spanning integer/single/double types, all justifications,
padded/trimmed decimals, fixed/scientific thresholds and scale factors -2, 0, 1,
2. A compact representative selection is retained in
`tests/fixtures/misclib-format-native.json`. Focused checks in
`tests/test_misclib_format.py` cover these outputs plus field failure, rounding
carry, large integers, subnormal values and extreme exponents. Native rounding-carry, omitted-exponent-letter and pre-trimming field-failure
examples were separately reproduced before documenting differences.

## Template pages and message printing

`compile_misclib_messages(source)` replaces the `format_specs` Perl-to-Fortran
code generator with immutable Python `MisclibMessage` pages. It accepts the
original text layout syntax:

```python
from mdanderson_stats import compile_misclib_messages, print_misclib_message

pages = compile_misclib_messages(
    ">>BEGIN Result\nPatient: %%%%%%%%\n>>CONTINUE\nStatus: %%%%\n>>END\n"
)
assert pages[0].render(["O'Brien"]) == "     Patient: O'Brien "
assert pages[1].render(["done"]) == "     Status: done"
```

Directives start in column one. `>>BEGIN [name]` starts a block; omitted names
become `message`, and explicit names are lowercased. `>>CONTINUE` finishes a page
and starts another with the same name; substitution numbering restarts on each
page. `>>END` closes the block. The returned tuple preserves source order and
repeated names. Each page exposes `name`, `template`, `substitution_widths` and
the directive's one-based `source_line`. Ordinary text outside blocks is ignored.
Unmatched/nested directives, empty pages with no lines, and unclosed blocks raise
`ValueError` rather than dropping text. Source text is limited to two million
characters. The original's unrelated continuation-line and Fortran variable-name
limits do not constrain Python output.

Trailing whitespace is removed from each source line; nonblank lines gain five
leading spaces. Blank lines remain blank, and existing indentation is preserved.
Each contiguous run of `%` is one fixed-width field. `render(substitutions)`
requires one string per field, truncates long values and blank-pads short ones,
as the source's `edit_format` does. Field width counts Python characters. Names
use ASCII letters, digits and underscores. Literal braces and quotation marks
remain literal; inserted values are never parsed as templates. No implicit
percent escape is introduced. Rendering returns text without an automatically
added final newline, and never edits the page in place.

`print_misclib_message(page, substitutions=(), *, console=None, force=False,
unit=None, unit_only=False)` routes the rendered text through `CDFConsole`.
Without a supplied console it creates one using the standard streams. The
console's existing `print_off`, `always_print`, `print_level` and `format_printed`
controls apply, as documented in [CDFLIB message controls](cdflib-message-format.md).
The console's prior template/substitution configuration is restored afterward;
`format_printed` reflects this attempt. Substitutions are validated before output,
even for a suppressed message. Streams remain caller-owned. Unit routing appends
one newline and respects `unit_only=False` by value, fixing the source's
presence-only interpretation. Forcing output skips optional help questions, an
intentional difference from the legacy prompt path.

Python pages replace **generated** Fortran FORMAT strings; arbitrary hand-written
Fortran FORMAT expressions are not interpreted. `render` replaces both
`edit_format` and `edit_message_format` for these pages; `print_misclib_message`
replaces the corresponding print calls. Screen clearing, pause/prompt functions,
file-unit allocation and the separate numeric-input helpers remain in the
remaining console-utility audit.

Native validation required two packaging repairs and explicit caller setup:

1. The archived `format_specs` fails Perl compilation because the assignment
   `$always_print = '.TRUE.'` lacks a semicolon. A private copy adds only that
   semicolon; the original download is unchanged.
2. Generated code says `USE print_it`, but the archive ships `print_it_mod`.
   The harness changes that module reference, supplies the user substitution
   assignments the generated code deliberately leaves blank, and initializes
   `print_off`/`num_subs` before use. In particular, named pages without fields
   otherwise inherit stale substitution state from earlier pages.
3. The original `print_it_mod.f90` is compiled unchanged with `-O2 -fcheck=all`.
   Three printed pages agree exactly, including continued pages, named pages,
   padding/truncation, apostrophes, double quotes, literal braces and blank lines.

`tests/fixtures/misclib-messages-native.json` records the source, substitutions,
expected output and repairs. Two focused tests check that comparison, repeat
rendering, literal substitutions, malformed structure, suppression/forcing,
stream routing and preservation of caller state.

## Interactive file selection

`misclib_open_file(*, console=None, message="", read_only=False,
appendable=True, delimiter="none", max_attempts=3, encoding="utf-8")` ports the
`open_file.f90` selection workflow using caller-owned Python text streams.

```python
from io import StringIO
from mdanderson_stats import CDFConsole, misclib_open_file

# An interactive call defaults to standard input/output:
# selection = misclib_open_file(read_only=True)
# The same workflow can be driven by explicit streams:
console = CDFConsole(StringIO("quit\n"), StringIO())
selection = misclib_open_file(console=console)
assert selection.status == 2 and selection.stream is None
```

The returned immutable `MisclibFileSelection` contains `status`, `stream`,
`path`, `action`, `delimiter` and optional `error`. Status meanings are 0 for
success, 1 for exhausted filename attempts, 2 for quit and 3 for back. On
success, use `with selection.stream as file:` after checking the status; the
caller owns and must close the stream. Failed/cancelled results have no stream
or path. EOF and underlying console-stream failures propagate, following the
existing `CDFConsole` contract; they do not fabricate an open-file result.

A filename is the first space-delimited word, preserving the source's inline
comment convention. Python also accepts single- or double-quoted filenames with
spaces, followed by an optional comment. Backslashes remain literal. `back` and
`quit` are case-insensitive commands. Blank names, NUL characters and unclosed
quotes are rejected within the bounded filename-attempt loop. Unicode filenames
are accepted without the legacy ASCII-warning dialog.

For an existing writable file, the user chooses `(q)uit`, `(r)etry`,
`(o)verwrite`, or `(a)ppend` when `appendable=True`. Read-only selection requires
an existing file. New writable paths select creation. Every selected action
then asks for quit, retry or proceed. Invalid action choices use the console's
bounded character-input policy. Open errors report context and allow another
filename attempt; exhaustion includes the last available error message.

**Confirmation occurs before opening, creating or truncating the file.** The
original truncates before its confirmation prompt and can delete the newly
opened/overwritten file when the user cancels. Python cancellation preserves
existing contents and creates no unwanted file. Confirmed creation uses exclusive
`x+` mode, so a file appearing between selection and creation is not overwritten.
Confirmed overwrite opens an existing file as `r+` and truncates it; append opens
an existing file as `r+` and seeks to its end. Append therefore sets the initial
position, as Fortran POSITION='APPEND' does; callers can deliberately seek later.
Read-only uses `r`, repairing the original's READWRITE access even in read-only
mode. A failure after opening closes the stream. Filesystem changes are not
transactional if an operating-system error occurs after truncation.

`delimiter` accepts case-insensitive `none`, `quote`, or `apostrophe`. The result
records this legacy list-directed formatting preference. Raw Python stream
writes remain literal: selecting a delimiter does not turn text into Fortran
list-directed or NAMELIST serialization. Python stream/descriptor management
replaces numeric Fortran unit allocation and the global already-open-unit check;
multiple handles follow normal Python/operating-system rules. Invalid arguments
raise before prompting, including unknown encodings.

`tests/test_misclib_files.py` exercises actual temporary files and scripted
console streams: create/read/append/overwrite, filenames containing spaces,
read-only permissions, cancellation before overwrite, retry before creation,
quit/back statuses, missing files, directory-open failures, bounded attempts and
EOF. These are filesystem integration checks against the inspected source
workflow and documented repairs, not claims of native Fortran transcript parity.
