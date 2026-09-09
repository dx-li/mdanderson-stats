# CDFLIB90 archive coverage

Catalog entry 21 is implemented with documented Python semantics. The
[completion audit](cdflib90-completion.md) reconciles source visibility,
documentation, alternate source and packaging; the
[machine-readable inventory](cdflib90-archive.json) links each interface to
implementation, native fixtures, independent tests and method notes.

The pinned archive contains **106 regular files**. Its inventory tool compares
all extracted files with archive bytes, retains hashes and rejects unknown roles.
Original native sources and executables are not bundled in the Python package.

| Archive material | Files | Disposition |
|---|---:|---|
| F95 distribution modules | 12 | All 48 public interfaces implemented and validated |
| F95 support modules | 7 | All public responsibilities mapped and validated |
| Binomial editor backup | 1 | Distinct source variant and false-success inversions reconciled |
| F95 build files | 2 | Replaced by Python packaging and CI |
| Legacy C implementations | 2 | All 24 distribution and 49 support names mapped |
| Legacy public C header | 1 | All 73 external declarations reconciled |
| Legacy Fortran source | 64 | All 66 routines/entries mapped, including two ENTRY statements |
| Documentation, installation and notices | 17 | Manuals reconciled, STATTAB cross-referenced and legalities retained |

## Distribution crosswalk

Each F95 distribution module explicitly exports four interfaces: its `cdf_*`,
`cum_*`, `ccum_*` and `inv_*` functions. The inventory checks all **48 names**.
The C/F77 libraries expose CDF solvers and paired-tail routines with older names.
All 24 legacy distribution names below are present in both source inventories.
The legacy routines do not introduce a thirteenth distribution. All twelve legacy
distribution contracts are implemented with separate C/F77 validation.

| Distribution | F95/Python suffix | Legacy CDF / tail names | Validated coverage |
|---|---|---|---|
| Beta | `beta` | `cdfbet`, `cumbet` | F95 and legacy C/F77 beta interfaces implemented |
| Binomial | `binomial` | `cdfbin`, `cumbin` | F95 and legacy C/F77 binomial interfaces implemented |
| Chi-square | `chisq` | `cdfchi`, `cumchi` | F95 and legacy C/F77 chi-square interfaces implemented |
| F | `f` | `cdff`, `cumf` | F95 and legacy C/F77 F interfaces implemented |
| Gamma | `gamma` | `cdfgam`, `cumgam` | F95 and legacy C/F77 gamma interfaces implemented |
| Noncentral chi-square | `nc_chisq` | `cdfchn`, `cumchn` | F95 and legacy C/F77 interfaces implemented |
| Noncentral F | `nc_f` | `cdffnc`, `cumfnc` | F95 and legacy C/F77 noncentral F interfaces implemented |
| Noncentral t | `nc_t` | `cdftnc`, `cumtnc` | F95 and signed legacy C/F77 interfaces implemented with df brackets |
| Negative binomial | `neg_binomial` | `cdfnbn`, `cumnbn` | F95 and legacy C/F77 interfaces implemented |
| Normal | `normal` | `cdfnor`, `cumnor` | F95 and legacy C/F77 normal interfaces implemented |
| Poisson | `poisson` | `cdfpoi`, `cumpoi` | F95 and legacy C/F77 Poisson interfaces implemented |
| Student's t | `t` | `cdft`, `cumt` | F95 and legacy C/F77 Student t interfaces implemented |

The [F95 method notes](cdflib90.md) and each legacy family's method document
explain domains, computed groups, endpoints, numerical repairs and failure behavior.
Legacy evidence is separate from F95 evidence. In particular, legacy F and
noncentral F include additional df inversions and wider bounds; their df searches
and noncentral t df searches require explicit sign-changing brackets when selecting
among multiple roots. Discrete families retain continuous extensions.

## Public F95 support

| Module | Public/support scope and status |
|---|---|
| `biomath_mathlib_mod` | [Five elementary helpers](cdflib-elementary.md), [four error/exponential helpers](cdflib-error-exponential.md), [seven gamma/digamma helpers](cdflib-gamma-support.md), [three gamma-ratio helpers](cdflib-gamma-ratios.md), [three beta/combinatorial helpers](cdflib-beta-support.md), the [gamma scaling factor](cdflib-gamma-factor.md), [two incomplete-gamma helpers](cdflib-incomplete-gamma.md), [two beta factors](cdflib-beta-factors.md), the [beta shape shift](cdflib-beta-shift.md), [fpser](cdflib-fpser.md), [apser](cdflib-apser.md), [bpser](cdflib-bpser.md), [bgrat](cdflib-bgrat.md), [basym](cdflib-basym.md), [bfrac](cdflib-bfrac.md), and [bratio](cdflib-bratio.md) complete all 35 mathematical procedures; imported [constants](cdflib-constants.md) are available through `cdflib_constants` |
| `zero_finder` | [Implemented](cdflib-root.md): direct/reverse interval and step solvers, setup and per-search state; corrected roots and tolerances; shared globals become result fields |
| `biomath_constants_mod` | [All 27 constants](cdflib-constants.md) implemented, with audited numeric values and documented legacy kind/unit identifiers |
| `cdf_aux_mod` | [Implemented](cdflib-aux.md): all 13 native descriptors, batch validators, complement/range helpers and root-state adapters with explicit Python success/error behavior |
| `biomath_interface_mod` | [Implemented](cdflib-console.md): all six numeric input overloads, character/string input, basic output, per-console streams and [all eight list-editing actions](cdflib-number-list.md); [array formatting](cdflib-array-format.md) and [message controls](cdflib-message-format.md) with documented Python templates |
| `biomath_sort_mod` | [Implemented](cdflib-sort.md): all four `sort_list` overloads and custom comparators |
| `biomath_strings_mod` | [Implemented](cdflib-strings.md): ASCII conversion and reentrant lexer `qlex` with documented repairs |

The source declaration crosswalk covers 121 support names in addition to the
48 distribution names. The completion verifier compiles all 19 unchanged F95
modules and imports 252 public names: those 169 names plus 83 transitive USE
associations. Python uses the documented support namespaces instead of copying
all implicit Fortran aliases. Private procedures and generic overload bodies are
implementation details; their public generic behavior is tested.

## Legacy support

All **49 support names** are exposed by `mdanderson_stats.dcdflib_support`:

- [Ten primitives](dcdflib-support.md): machine parameters, polynomial evaluation
  and the seven C translation helpers.
- [31 mathematical helpers](dcdflib-math.md): separate C/F77 validation, including
  the distinct legacy exponential-limit convention.
- [Three normal/t quantile helpers](dcdflib-quantile-helpers.md): starting formulas
  and refined normal inversion.
- [Incomplete-gamma inversion](dcdflib-gamma-inverse.md): starting hints,
  representability checks and independently verified extreme-tail repairs.
- [Four root-finder interfaces](dcdflib-root.md): separate search state,
  reverse communication, failure flags and the native stopping convention.

Together with the 24 distribution names, these account for every external C
function. The 66 F77 names correspond after the `erf`/`erf1` and `gamma`/`Xgamm`
aliases and exclusion of seven C-only helpers. The two static C functions are
private implementation details. Native process termination becomes a Python
exception; native status/result conventions and numerical defects are documented
rather than silently copied.

## Alternate binomial source

`#cdf_binomial_mod.f90#` differs from the primary `cdf_binomial_mod.f90`, including
formatting and IMPLICIT NONE declarations. It is not listed among the Makefile's
19 F95 source inputs. Both hashes and declaration inventories are retained.
The [binomial reference tool](../tools/reference_cdflib_binomial.py) compiles both
versions with unchanged source bytes and captures 102 cases each. All 25 backup
probability inversions in this fixture return success without replacing the
placeholder input probability. Source inspection confirms assignment/termination
ordering differences. Python follows the shared statistical definition and
repairs the inverse behavior; the Makefile-selected primary source remains the
reference version. This alternate source is reconciled, not counted as a separate
product or assumed byte-identical.

## Reproduction and limits

Run `uv run python tools/audit_cdflib90.py`, then
`uv run python tools/audit_cdflib90_completion.py` from the repository root.
The pinned archive and extraction under ignored `research/raw` are required;
missing or changed source fails the audit. The second command additionally
requires gfortran and records compiler details and the import-driver hash.

Export compilation proves public visibility, not numerical accuracy. Per-interface
tests provide native comparisons and independent mathematical or protocol checks;
the method documents retain representability limits, explicit errors and benchmark
scope. No uniform native-code speedup or bitwise equivalence is claimed.

All 17 documentation members have dispositions in the completion evidence.
The embedded STATTAB 2.0 manual belongs to the separate pending
[STATTAB entry 23](stattab-research.md), whose workflow remains in the catalog goal.
