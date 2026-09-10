# Misclib: shared numerical library and scalar maximization

Misclib is a library of statistical-software support routines by the MD Anderson
Section of Computer Science, Department of Biomathematics. This port reuses the
existing CDFLIB implementations where the archived procedure bodies agree.
Catalog entry 87 remains **partial**: its numerical methods are available, while
some formatting, file/prompt and matrix-permutation utilities still need ports
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
| `sort_mod`, `sort_permutation_mod` | Existing `sort_list` offers list sorting, but source blocks differ; matrix sorting, forward/inverse permutations and callback contracts still need audit/implementation |
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
