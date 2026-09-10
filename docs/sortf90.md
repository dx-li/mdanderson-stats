# SORTF90: alphabetize Fortran program units

`sortf90(source)` covers the source-organizing operation in Barry W. Brown's
SORTF90 utility. This catalog entry is a programming utility, not a statistical
estimator. It alphabetizes top-level named units and recursively alphabetizes
contained procedures, case-insensitively, preserving the spelling and body of
each unit. The default first-level separator contains 69 asterisks; deeper
separators contain 69 periods, matching the executable Perl source.

```python
from pathlib import Path
from mdanderson_stats import sortf90

source = Path("model.f90").read_text()
organized = sortf90(source)
# Write to a separate file for review before replacing the original.
Path("model.sorted.f90").write_text(organized)
```

The function does no file I/O, creates no temporary directories or automatic
backups, and never overwrites the input. Pass `separators=(first, deeper)` to
customize separators, including their newlines. Empty strings disable them.

Like the original, this is a deliberately restricted source organizer, not a
full Fortran parser. Supported units are named PROGRAM, MODULE, SUBROUTINE and
FUNCTION, including typed/prefixed procedures and nested CONTAINS sections.
Openings, CONTAINS and explicit END statements must each occupy one complete
line. END kind and optional name must match the opening. Interface blocks inside
unit bodies are retained without sorting their declarations. Use preprocessed,
free-form source; fixed-form, derived-type definitions, submodules, bare END,
and multiline or semicolon-separated structural statements are unsupported.
Unsupported constructs detected by the organizer raise ValueError; compiler
validation remains necessary because this restricted parser cannot diagnose
all invalid Fortran. Top-level units are also sorted, so separate modules whose
compilation depends on source order should be organized separately.

**The original intentionally discards comments and blank lines outside unit
bodies:** before the first unit, after CONTAINS before its first procedure,
between completed procedures and after the last unit. This function retains
that behavior, strips trailing spaces and normalizes line endings to LF. Keep
an original copy and review the returned text, especially license headers and
comments documenting the following procedure. Comments within bodies remain.

Unlike the Perl program, duplicate sibling names, mismatched endings and
unterminated units raise an error instead of potentially losing or corrupting
routines. Keywords inside string literals do not open or close units.

## Validation and provenance

Two independently written source examples were passed through the **unchanged
original Perl executable**. Their exact outputs are stored in
`tests/fixtures/sortf90-native.json`; they cover top-level sorting, nested
sorting, typed/recursive procedures, separators, discarded inter-unit comments,
and opaque interfaces. Focused tests check byte-for-byte agreement, idempotence,
string literals and invalid structure. Both organized examples also passed
`gfortran -fsyntax-only`. No numerical test or additional CI job is applicable
to this text utility.

Source: [MD Anderson SORTF90 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SORTF90/SORTF90_V1.tar.gz),
containing `readme`, `sortf90.routines` and `sortf90.routines.doc`.
The executable uses periods for deeper separators, although a prose passage
says hyphens. Hashes are in `sortf90-sources.json`. Original source files are not
redistributed. The archive identifies its author but provides no separate
software license grant.
