# FLECS90 conversion and native-reference audit

October 8, 2026. Catalog entry 43 moved from pending to implemented after
recovering the source-defined translator and validating its Python workflow.
Coverage is now 89 implemented, 42 partial and seven pending among 138 entries.
This changes one catalog workflow, not a percentage of statistical methods.

## Primary source and license

The catalog detail endpoint returned HTTP 500, but the corresponding archive
at `SoftwareDownload/SoftwareFiles/FLECS90/FLECS90_V1.tar.gz` was retrievable
from the institutional HTTPS host. Its 35,436 bytes have SHA-256
`82eb73ef0b6e8e27c850fae8a9d1fc5b7b8e1a4505a729e0476267e9efc39c29`.
The archive contains C source, a manual and README. The latter says “This code
is placed in the public domain. Enjoy.” Its complete bytes are retained in
`notices/mdanderson-flecs90-readme.txt`; original source remains ignored research
input. `docs/flecs90-source.json` records every source-file checksum.

Unlike the earlier secondary-source lead, these files define all native
alternation, repetition and procedure translations, fixed-form reading,
continuations, labels, program-unit resets and output options. FLECS90 is
itself a source conversion utility; its Python implementation emits Fortran,
which is its cataloged purpose. It does not replace numerical Fortran methods
with a runtime wrapper or claim a compiler implementation.

## Executed native and compiler references

The reference executable uses GCC 14.2.0, `-std=c99 -fcommon -O0`.
The build renames its local `getline` to avoid the modern stdio signature
collision. It initializes `work.c`'s `flecs_in` pointer to NULL and
`qotherwise` flag to zero, removing undefined initialization behavior.
These documented harness changes leave keyword dispatch and translations
unchanged. The exact archived source is kept separately from the build copies.

`tools/reference_flecs90.py --check` independently reproduces all 144 committed
translations from C, without consulting the Python translator for expected
outputs. Nine input programs cover inline/block IF and UNLESS, WHEN/ELSE,
CONDITIONAL/CASE/SELECT/OTHERWISE, counted and pre-/post-test loops, LOOP,
CYCLE/EXIT, procedures/REVERT, original Fortran control constructs,
continuations, statement labels and multiple program units. Every combination
of comment, line-number, SELECT CASE and loop-name options is exercised.
Python output matches every reference byte for byte.

GNU Fortran 14.2.0 compiled all 144 Python output programs. The 136 terminating
variants executed with independently stated expected outputs: 5 for the basic
alternation program, 40 for all loop families, 33 for branch selection, 4 for
procedures, and 2 for the continuation and named-loop fixtures. Other programs
terminate without printed output. The eight unlabelled variants of the nested
original-Fortran-DO fixture were not run: their EXIT ends the inner DO, leaving
an infinite outer LOOP, exactly the behavior warned about by the native manual.
They remain compiled, not executed, outcomes.

Two additional compiled programs verified preserved quoted spaces/literal
parentheses and EXIT inside a finished inner counted loop. The latter must
target that inner loop rather than the surrounding FLECS loop. No undefined
native output is asserted for these corrected cases. The temporary compiler
was extracted locally from Debian packages obtained through a signature-verified
package index; no compiler is required by the Python package at runtime.

## Functional Python completion and explicit differences

`flecs90.py` supplies bounded translation and line provenance.
`translate_flecs90_file` atomically saves a separate output and defaults to
no-clobber; input aliases and existing outputs are protected. The CLI supports
multiple files, all four output options, check-only operation and nonzero
error outcomes. Pure Python is sufficient for this workflow.

Native error recovery may discard source, close unfinished structures or return
exit status zero after errors. The Python API instead raises line-specific
`FLECS90SyntaxError` and does not save malformed output. Quoted string whitespace
and parentheses are preserved; ordinary Fortran IF/ELSE does not dereference a
missing FLECS stack; finished-loop transfers use the correct generated loop;
empty conditionals do not emit unmatched END IF. These are intentional
corrections. General Fortran parsing, compilation and byte-identical native
error recovery are outside this translator's completion criterion.

166 focused tests passed with warnings treated as errors, including all 144
references, diagnostics, bounds, corrections, line mapping, file protection and
CLI checks. Status promotion is supported by the recovered primary source and
this complete, executable workflow; native undefined behavior is not required
for coverage.
