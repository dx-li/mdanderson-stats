# Misclib coverage audit

Catalog entry 87 is implemented with the Python interfaces and compatibility
choices described in [Misclib methods](misclib.md). All advertised families and
all executable source modules have mappings. This status describes usable Python
versions; it does not promise identical Fortran unit numbers, compiler-generated
code, error messages, unsafe buffer behavior or platform-specific I/O semantics.

The final audit compares every one of the **20 regular archive files** against
its original ZIP member bytes. `misclib-sources.json` records source provenance;
`misclib-final-audit.json` records hashes, additional procedure comparisons and
all list-sort token differences. The earlier `misclib-shared-source.json` retains
the mathematical/root/string comparison. The original archive is not bundled.

| Source / responsibility | Implementation and evidence |
|---|---|
| `constants_mod.f90` | All constants in `cdflib_constants`; full normalized module matches CDFLIB90. Kind codes and logical units remain legacy identifiers, not Python handles. Existing [constant evidence](cdflib-constants.md) applies |
| `mathlib_mod.f90` | All 35 procedures match the audited CDFLIB90 source; reused numerical implementations and native/independent validation listed in [CDFLIB coverage](cdflib90-coverage.md) |
| `zero_finder_mod.f90` | Eight procedure blocks match CDFLIB90. Reentrant interval/step and direct/reverse APIs, with documented safeguarded interpolation/bisection replacing native TOMS 748; [root methods](cdflib-root.md) |
| `strings_mod.f90` | Five procedure blocks match CDFLIB90; ASCII case conversion and reentrant `qlex`, including documented lexical/error repairs; [string evidence](cdflib-strings.md) |
| `sort_mod.f90` | Four list overloads reuse `sort_list`. Differences from CDFLIB are names, diagnostics and equivalent declarations, not sorting logic. Matrix sorting uses `sort_matrix`; native numeric/character evidence in `tests/test_misclib_sort.py` |
| `sort_permutation_mod.f90` | `permutation_sort_matrix` and `permute_matrix`, including native gather orientation, negative-option reversal, partial blocks and custom callbacks; same matrix evidence |
| `max_fun_mod.f90` | `set_fun_max`, `fun_max`, `rc_fun_max`, per-search state and numerical safeguards; six compiled-native scenarios plus analytic and extreme-scale checks |
| `format_number_mod.f90` | `format_number`, integer/single/double representation, alignment, fixed/exponential thresholds, scale, trimming and fit. 240 native scenarios compared, with documented rounding/buffer/exponent repairs |
| `format_specs` | `compile_misclib_messages`: named/unnamed BEGIN/CONTINUE/END pages and fixed-width fields. Three native rendered pages after documented private semicolon/module-name repairs |
| `print_it_mod.f90` | `MisclibMessage.render` and `print_misclib_message` replace generated-format editing/printing. `CDFConsole` supplies clear-screen, hold, prompt, choices and y/n; ASCII case functions reuse string helpers. OS-managed streams replace `igtfun` unit-number allocation |
| `get_values_from_user_mod.f90` | Ten numeric/character/string/list procedures match CDFLIB's interface module, and four case procedures match its strings module; reuse `CDFConsole`, `CDFNumberList` and string functions with their validated Python semantics |
| `interface_mod.f90` | Twenty procedure blocks match CDFLIB90. `CDFConsole.pause` covers the one additional routine, verified against its unchanged compiled body. Message substitution arrays become explicit Python templates/arguments; report-unit state becomes a caller-owned report stream |
| `open_file.f90` | `misclib_open_file` with explicit result/status, read/create/append/overwrite, quit/back/retry and bounded attempts. Real-filesystem tests verify cancellation and read-only repairs |
| Three `DOC/misclib` files | Equivalent TeX/PDF/PostScript documentation; method requirements reconciled with executable behavior, including erf normalization typo and permutation-description discrepancies |
| Two installation texts and `COMPILE.IT` | Fortran build/install workflow replaced by the existing Python package and wheel build |
| `LEGALITIES.txt` | Original notice preserved in `notices/mdanderson-misclib-Legal.txt`; third-party terms and implementation provenance retained |

## Console mappings and final gaps closed

`CDFConsole.get_numbers` covers scalar/vector integer, single and double input;
`get_character`, `get_yn`, `get_string` and `get_list_double` cover choices,
commented strings and interactive list editing. Their source blocks agree after
removing nonbehavioral declarations/module prefixes. EOF, invalid values and
exhausted retries follow the existing explicit Python error contract. Source
message FORMAT strings become Python message text/templates, not a general
Fortran FORMAT interpreter. `write_array` retains the existing supported numeric
format API documented in [array formatting](cdflib-array-format.md).

The final two additions are `CDFConsole.pause()` and per-console
`clear_screen_before_print=False`, `window_size=24`. Pause reproduces the native
message and consumes one input record. Clear-before-print writes the configured
number of blank lines to console output only when a message will be displayed,
including for report-only output, as the source does. Suppressed messages do not
clear. `hold()` remains available with its original distinct message.

All source storage controls are represented by Python data or documented runtime
choices: immutable message pages/substitution widths replace positional scratch
arrays, explicit substitutions replace saved strings, and caller-owned streams
replace logical-unit allocation. Legacy Fortran list-directed/NAMELIST delimiter
selection is retained as file-result metadata; literal Python writes are not
silently reserialized. There is no global registry forbidding multiple handles
to a path. These differences are explicit in [file selection](misclib.md).

Validation at completion: 83 affected tests passed, comprising all Misclib
modules plus existing CDFLIB console/message checks. Shared numerical functions
were not changed and their historical suites were not rerun. Native comparison
and source-identity evidence are reused instead of adding duplicate tests. No CI
jobs or dependencies were added.
