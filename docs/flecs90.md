# FLECS90: FLECS-to-Fortran translation in Python

Catalog entry **43** is implemented as a Python source translator. FLECS90's
native purpose is to convert the FLECS control-language extensions to fixed-form
Fortran 90; it is not itself a statistical method. The Python implementation
performs that translation without a C compiler or executable at runtime.

## Text and file workflow

```python
from mdanderson_stats import translate_flecs90

source = """      PROGRAM EXAMPLE
      INTEGER I,TOTAL
      TOTAL=0
      FOR (I=1,3)
      TOTAL=TOTAL+I
      FIN
      PRINT *,TOTAL
      END
"""
result = translate_flecs90(source)
print(result.fortran_source)
# FLECS_LOOP_1: DO I=1,3 ... END DO FLECS_LOOP_1
# Compiling and executing this Fortran program prints 6.
```

`translate_flecs90_file("example.flx")` saves `example.f` atomically and returns
the same `FLECS90Translation` object. It preserves the input and refuses to
replace an existing output unless `overwrite=True`. Input/output aliases are
rejected. `source_line_numbers` maps each generated line to its original
physical input line, including generated continuation cards.

The command-line workflow accepts multiple input files:

```sh
uv run python -m mdanderson_stats.flecs90_cli example.flx
uv run python -m mdanderson_stats.flecs90_cli -f example.flx
uv run python -m mdanderson_stats.flecs90_cli -c -n -s --overwrite example.flx
```

`-f` checks syntax without creating output. `-c` echoes comments/blank lines,
`-n` includes source numbers in columns 74–78, `-s` uses Fortran `SELECT CASE`,
and `-l` suppresses generated loop names. File errors and malformed FLECS
structures produce a nonzero CLI exit status.

## Covered native language

| FLECS construct | Fortran translation |
| --- | --- |
| IF, UNLESS | Inline logical IF or block IF with FIN |
| WHEN ... ELSE | Block IF/ELSE; inline and multiline arms |
| CONDITIONAL, CASE without a selector | Ordered logical branches and OTHERWISE |
| SELECT, CASE with a selector | Ordered equality IF branches, or SELECT CASE with `select_case=True` |
| DO/FOR with parenthesized bounds | Named counted DO loop |
| WHILE, UNTIL | Pre-test DO WHILE |
| REPEAT WHILE, REPEAT UNTIL | DO with a post-body IF/EXIT test |
| LOOP, CYCLE, EXIT | Indefinite loop and transfers to the enclosing FLECS loop |
| TO, hyphenated procedure calls, REVERT | CONTAINS/internal subroutines, CALL and RETURN |
| FIN, original Fortran IF THEN/ELSE/ENDIF, bare END | Structure completion and program-unit boundaries |

Existing Fortran statements, labels and Fortran continuation cards pass through.
FLECS logical statements can continue using a nonblank/nonzero column 6.
Input uses fixed-form columns 1–72; later columns are sequence fields and are
ignored. Comment cards begin with C/c or `*` in column 1. Generated lines wrap
at column 72. Bare END resets procedure and loop numbering for the next unit.

The native `-s` option changes which selector expressions Fortran accepts:
SELECT CASE requires integer/logical/character selectors and compile-time
case constants. The default equality-IF translation allows runtime values.
With `label_loops=False` (`-l`), EXIT/CYCLE apply to the innermost Fortran loop,
including original Fortran loops, so behavior can differ from FLECS. Default
named loops preserve the FLECS target. A Fortran compiler remains responsible
for validating the untouched Fortran and type-checking expressions.

## Validation, corrections and boundaries

The recovered institutional archive declares its source public domain; its
[original README and warranty notice](../notices/mdanderson-flecs90-readme.txt)
are retained. [Source provenance](flecs90-source.json) pins its SHA-256 hash.

Nine source programs exercise all native structure families. All 16 option
combinations match 144 C-reference output files exactly. GNU Fortran 14.2.0
compiled all 144 generated files. Of those, 136 programs executed with their
expected output. The eight unlabelled-loop variants containing an original
Fortran DO inside a FLECS LOOP were deliberately not executed: their documented
EXIT semantics make this particular fixture loop indefinitely. Two additional
programs executed to verify quoted-text preservation and a single-statement
inner loop's EXIT target. See the [reference audit](../research/flecs90-audit.md).

The Python implementation preserves whitespace and parentheses inside quoted
strings, handles ordinary Fortran IF/ELSE without dereferencing an absent FLECS
stack, and gives a finished loop's inline EXIT/CYCLE its own loop target.
Empty conditionals produce no unmatched END IF. These are explicit corrections,
not claims to reproduce undefined C behavior. Malformed structures raise
`FLECS90SyntaxError` with a physical line number instead of silently repairing
or discarding statements according to the native error-recovery scheme.
Native terminal-keyword continuation cards are ignored as in the original.
This does not implement a general Fortran parser, compiler or interpreter.

Inputs are bounded to 4 MiB of ASCII text and 99,999 physical lines, logical
FLECS statements to 32,768 characters, nesting to 512 structures, and generated
output to 16 MiB. Python tests cover the native outputs, corrections, line
mapping, diagnostics, bounds, atomic file handling and CLI behavior.
