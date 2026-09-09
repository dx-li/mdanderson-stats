# STATTAB shared-source reconciliation

The [reproducible source audit](stattab-shared-source.json) reviews all nineteen
modules shared with CDFLIB90 and compiles **259 public imports** against STATTAB's
unchanged source: 252 shared imports and seven application-specific imports.
Run `uv run python tools/audit_stattab_shared.py` with both pinned archives present
and gfortran installed. The tool checks archive hashes and builds in a temporary
directory. The [completion audit](stattab-completion.md) incorporates this review
and reconciles the complete application/manual contract.

## What changed between archives

Two modules, `biomath_mathlib_mod` and `cdf_gamma_mod`, are byte-identical.
Four more have equal logical statements after removing comments, whitespace,
case differences, standalone `PUBLIC`, and `IMPLICIT NONE`: constants, beta,
normal and Poisson. The other thirteen modules have individually reviewed diffs
in the JSON report.

Except for binomial chance inversion, the reviewed changes relocate declarations,
express visibility through inline attributes, omit redundant intrinsic declarations,
or change equivalent character/format syntax. The old and new blank-input warning
formats were compiled and executed; their complete output, including the leading
blank line, agrees. No numerical algorithm change was found in these other modules.

The comparison is a review aid, not a general Fortran parser or a proof of
numerical equivalence. Compilation proves that the listed imports are accessible;
it does not discover every export or prove the procedures correct. Existing
[CDFLIB90 validation](cdflib90-completion.md), STATTAB's native-session evidence,
and independent Python tests provide the numerical and behavioral evidence.
Inherited native defects remain subject to the documented Python repairs.

## Binomial inverse-probability defect

STATTAB moves the final assignments of `pr` and `cpr` inside the fourth of four
inverse branches. The other three branches therefore leave their `INTENT(OUT)`
arguments undefined even when status is zero. The probe initializes these outputs
to sentinels for observation; sentinel retention is compiler-specific evidence,
not a valid Fortran output contract or a portable expected result.

All cases use four trials. Targets come from the finite binomial sum, independently
of either implementation:

| Successes | Correct chance | Cumulative target | Native branch | Observed native outputs |
|---|---:|---:|---|---|
| 0 | 0.25 | 0.31640625 | Match lower tail, vary chance | Both sentinels retained |
| 0 | 0.75 | 0.00390625 | Match lower tail, vary complement | Both sentinels retained |
| 3 | 0.25 | 0.99609375 | Match upper tail, vary chance | Both sentinels retained |
| 3 | 0.75 | 0.68359375 | Match upper tail, vary complement | Approximately 0.75 and 0.25 |

Python's existing beta inversion returns the correct chance and complement for
all four cases. Eight regression cases in `test_stattab_results.py` supply each
case through either `cum` or `ccum` and compare against the independent target.
The fourth native branch also exits before rebuilding its complementary iterate;
the observed accurate result for this case does not establish correctness for
other inputs. No native undefined value is used as a numerical oracle.

## Shared public interfaces

The 252 imports comprise 169 directly declared names and 83 names exposed through
Fortran `USE` association. The existing [CDFLIB90 mapping](cdflib90-completion.json)
provides the explicit names and transitive paths; the new audit recompiles every
one against STATTAB's archive. Python replacements retain their documented
contracts rather than introducing a second copy of the shared implementation.

| Native responsibility | Python implementation |
|---|---|
| Twelve distribution modules | Existing `cdf_*`, `cum_*`, `ccum_*`, and `inv_*` interfaces; `stattab_solve` adds the application selector and columns |
| `biomath_constants_mod` | `cdflib_constants` |
| `biomath_mathlib_mod` | Validated CDFLIB90 mathematical support functions |
| `biomath_sort_mod` | `sort_list` |
| `biomath_strings_mod` | Case conversion and `qlex` |
| `biomath_interface_mod` | `CDFConsole`, `CDFNumberList`, `CDFConsoleError`, and array formatting; explicit object/stream state replaces global units and print state |
| `zero_finder` | `ZeroFinder`, `ZeroFinderResult`, and direct/reverse-communication functions; explicit state replaces mutable global results |
| `cdf_aux_mod` | `cdflib_aux`, `CDFParameter`, and `CDFDistribution` |

## Application-specific public interfaces

`biomath_file_io_mod` declares only `open_file` and `report_file_dialogue` public.
They map to `stattab_open_file` and `stattab_report_file_dialogue`; the
[console documentation](stattab-console.md) records stream ownership, cancellation,
read-only access, confirmation before overwrite, and replacements for Fortran
unit allocation and list-directed delimiters.

`stattab_aux_mod` exposes five names: `ndist`, `one_parameter`, `the_distribution`,
`distributions`, and `display_banner`. Its twelve `d1`–`d12` constants are explicitly
private. The descriptor types are distinct from CDFLIB90's similarly named types.
Python represents the menu collection and its size through `STATTAB_DISTRIBUTIONS`;
`STATTABDistribution` holds parameter names and computed groups, while
`stattab_help` and result formatting supply descriptions and headings. The console
supplies the banner. These are semantic replacements for the source records,
not aliases claiming Fortran layout compatibility. All seven application-specific
names compile in the export probe.

The main program is an application entry point rather than an importable module.
Its request parsing, dispatch, help, numerical postprocessing, and status handling
map to the implemented session, result, reporting, and console layers. The [completion audit](stattab-completion.md) reconciles the complete application
contract and historical documentation.
