# SPPCR batch input and validated experiment data

`parse_sppcr_batch` reads the original ordered batch format into immutable
`SPPCRData`. `format_sppcr_batch` writes canonical records, and `sppcr_data` builds
validated experiment data directly. FileMaker input, interactive input, historical
RNG reconciliation and the complete reporting/application workflow remain pending.

```python
import numpy as np
from mdanderson_stats import (
    parse_sppcr_batch,
    sppcr_bootstrap,
    sppcr_bootstrap_intervals,
)

text = """nallele 3
nrun 2
nwell 20 30
allelesizes 100 102 104
progenitor 100 102
run 0.5 4 8 2
run 1 10 15 5
"""
data = parse_sppcr_batch(text)
bootstrap = sppcr_bootstrap(
    data.dna,
    data.seen,
    data.wells,
    progenitor=data.progenitor,
    rng=np.random.default_rng(22),
    replicates=1000,
)
limits = sppcr_bootstrap_intervals(bootstrap)
```

## Units and identities

Native `values_to_structures` interprets file DNA as genome equivalents and
multiplies it by two to obtain allele equivalents for the likelihood calculation.
The Python object retains both `genome_dna` (file units) and `dna` (twice that
value). Pass `data.dna` to the numerical APIs, which perform no further conversion.
The formatter writes `genome_dna`, so repeated parse/format cycles do not double
DNA amounts again. Overflow of the conversion raises ArithmeticError.

The object also holds `wells`, `seen`, `allele_sizes`, `progenitor_sizes`, the
zero-based `progenitor` index pair, and `omitted_allele_sizes`. Allele order is
preserved. Sizes must be distinct positive integers; both progenitor sizes must
appear among the supplied alleles. Repeat a progenitor size for a homozygote.

`unseen_alleles="drop"` is the default, following native input conversion: columns
with no detections at any active DNA level are removed and their labels recorded.
If this would remove a progenitor, Python raises a clear error instead of retaining
the source's invalid index −1. Pass `unseen_alleles="retain"` to keep every supplied
allele, including a never-seen progenitor or an entirely zero experiment. The
existing fitting and summary APIs then expose their usual boundary diagnostics.
Removal depends only on the current experiment, never stale array storage.

`sppcr_data(genome_dna, seen, wells, allele_sizes, progenitor_sizes,
*, unseen_alleles="drop")` provides the same validation and conversion without text
parsing. It accepts one experiment: `seen` has shape `(levels, alleles)`, DNA is a
positive vector and wells broadcast to the level axis. Counts must be integers
below `2**53`, wells positive and seen no greater than wells. Output arrays own
immutable storage independent of input arrays. The constructor function validates;
the result dataclass is intended as an output container.

## Batch grammar

Records must appear in this order:

1. `nallele` followed by one integer allele count;
2. `nrun` followed by one integer DNA-level count;
3. `nwell` followed by one integer per DNA level;
4. `allelesizes` followed by one integer size per allele;
5. `progenitor` followed by two integer sizes;
6. exactly `nrun` records beginning with `run`, a DNA amount and one integer
   detection count per allele.

Keywords are case-insensitive. Blank lines are ignored. `/` or `#` starts a
comment extending to the end of the line. Integer lists can continue onto later
nonblank, uncommented lines; there must be no extra tokens after the last required
value on its line. The DNA amount must follow `run` on the same physical line.
DNA accepts decimal and E/D exponent notation. Integer fields require integer
syntax and signed-int32 representability, matching the source's integer storage;
`20.0` is not an integer token. Negative counts are rejected, not converted to
positive values. Punctuation such as `=`, commas and colons is not silently erased.
Tabs are explicitly accepted as whitespace, extending the source's lexical rules.

Parsing defaults to a maximum of 50 alleles and 50 runs, matching the native
fixed arrays. `max_alleles` and `max_runs` may be increased explicitly for Python;
`max_characters` defaults to one million. Invalid limits, oversized input,
incomplete records, duplicates, wrong ordering, extra records, malformed numbers,
invalid identities and inconsistent counts raise exceptions. Syntax errors include
a line number. The parser does not print errors and continue with partial state.
It reads text supplied by the caller; no files are opened implicitly.

`format_sppcr_batch(data)` writes the retained columns in canonical records with
lines no longer than 240 characters, safely below the native 256-character input
buffer. Integer lists wrap onto continuation lines. Dropped zero columns and
original comments/layout are not reconstructed. Values exceeding signed-int32
range cannot be formatted for the native parser and raise ValueError. Python
experiments exceeding 50 runs/alleles require explicitly increased limits when
parsed again and are not compatible with native fixed-size arrays.

## Source evidence and validation

`tools/reference_sppcr_batch.py` compiles the archived sources and invokes the
actual batch reader and input conversion. The probe explicitly zeros unused
input storage for reproducibility; a separate case injects one stale row. The
fixture retains the pinned archive hash, driver, input text, exit codes and raw
outputs. Native evidence confirms ordinary files, uppercase keywords, comments,
continuation lines, exponent DNA, dropping unseen nonprogenitor alleles, unit
conversion and identity mapping.

Defect probes show that the native lexer turns a negative count into a positive
count, an unseen progenitor becomes index −1, and an unused stale row prevents
removal of an otherwise unseen allele. These outputs document defects and are not
Python compatibility targets. The existing checked CDFLIB lexer is reused for
Python tokenization; numeric sign handling and validation remain explicit.

Tests compare valid native outputs, exercise the repaired cases, round-trip
canonical records including long lists, verify DNA conversion exactly once,
check invalid inputs and limits, and run parsed data through bootstrap fitting
and confidence intervals. Parsing is linear in input size; data validation uses
array operations. No native-versus-Python speed claim is made for this parser.
