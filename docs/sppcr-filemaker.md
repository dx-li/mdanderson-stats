# SPPCR FileMaker-style numeric exports

`parse_sppcr_filemaker` reads the numeric row format accepted by the source's
`read_mjs_one` routine. `format_sppcr_filemaker` writes canonical quoted CSV rows.
Both use the validated `SPPCRData` model, including explicit DNA units and allele
identity mapping. This is a text-file format interface, not a live FileMaker
application connection. Interactive input, historical RNG reconciliation and the
complete reporting/application workflow remain outstanding.

```python
import numpy as np
from mdanderson_stats import (
    parse_sppcr_filemaker,
    format_sppcr_filemaker,
    sppcr_bootstrap,
    sppcr_bootstrap_intervals,
)

text = """100 102 20 .5 100 4 102 8 104 2
100 102 30 1 100 10 102 15 104 5
"""
data = parse_sppcr_filemaker(text)
canonical_csv = format_sppcr_filemaker(data)
bootstrap = sppcr_bootstrap(
    data.dna,
    data.seen,
    data.wells,
    progenitor=data.progenitor,
    rng=np.random.default_rng(82),
    replicates=1000,
)
limits = sppcr_bootstrap_intervals(bootstrap)
```

## Row contract

Each row describes one DNA level, with fields in this order:

1. first progenitor allele size;
2. second progenitor allele size;
3. number of wells;
4. DNA amount in genome equivalents;
5. repeated allele-size, detection-count pairs, one pair for every allele.

There is no header record. Every row must repeat the same ordered progenitor pair
and the same ordered allele sizes. Well counts and DNA amounts may vary by row.
Repeat a progenitor size for a homozygote. The parser accepts whitespace-separated
numeric rows and numeric CSV with optional quoted fields. Decimal and E/D exponent
notation are accepted for DNA; all other fields require signed-int32 integer
syntax. `/` or `#` begins an end-of-line comment. As explicit extensions, tabs
are whitespace and blank/comment-only rows are ignored.

CSV fields must contain exactly one numeric token. Empty fields, extra delimiters,
unclosed quotes, compound fields, odd allele/count lists, changed identities,
nonfinite or underflowed DNA values and invalid counts raise ValueError. Unlike
the native blanket removal of quotes, commas and non-ASCII characters, the Python
reader preserves field boundaries and rejects malformed content. It never shifts
later values into a missing field or prints an error and resumes with partial data.
Syntax and cross-row identity errors include the source line number.

The default maximum is 25 alleles: the native reader allows 54 values per row,
and four header fields leave room for 25 pairs. The default maximum of 50 runs
matches native storage. Python callers may explicitly increase `max_alleles` or
`max_runs`; `max_characters` defaults to one million. Python does not silently
truncate physical lines to the source's 500-character input buffer. Larger Python
inputs are not promised to work in the native application.

## Shared data semantics

The parser returns the same [experiment model](sppcr-batch.md) as batch input.
`genome_dna` retains file units; `dna` is twice that value and is ready for fitting.
Allele sizes must be distinct and positive, wells positive, counts nonnegative and
no greater than wells, and progenitor sizes present among the alleles.

The default `unseen_alleles="drop"` removes never-seen nonprogenitor columns and
records their labels in `omitted_allele_sizes`. Dropping a never-seen progenitor
raises instead of exposing the native invalid index. Use `unseen_alleles="retain"`
to keep all columns, including an unobserved progenitor or all-zero experiment.
Output arrays have immutable storage independent of inputs; progenitor indices
are zero-based and refer to the retained allele order.

## Canonical output

The formatter emits one quoted numeric CSV row per DNA level with a trailing
newline. It writes genome DNA units, retained allele columns and the same ordered
progenitor sizes on every row. It does not reconstruct dropped columns, comments,
original quoting or layout. Parse/format cycles preserve units, counts and identity.

For compatibility with the original reader, output must fit 25 alleles, 50 runs,
signed-int32 integer values and 500 characters per row. The formatter validates
these limits and raises instead of splitting a row or allowing native truncation.
A record with large integer labels/counts may exceed 500 characters even with
25 or fewer alleles. The caller owns filesystem access; these functions consume
and return text without opening files.

## Validation

`tools/reference_sppcr_filemaker.py` compiles the original source and invokes
`read_mjs_one` and its input conversion with controlled zeroed unused storage.
The fixture retains the archive hash, driver, input records, exit status and raw
output, including the source's diagnostic line echoes. Native comparisons cover
plain numeric rows, quoted CSV, exponent DNA, comments, homozygous progenitors,
unit conversion and dropping unseen nonprogenitor alleles.

Tests also check canonical round trips, retained unseen progenitors, parsing into
bootstrap fits and confidence intervals, missing/malformed fields, identity changes,
invalid counts and numeric types, input limits, native formatter limits, and
explicitly increased Python run limits. Parsing is linear in input size; shared
experiment validation uses array operations. No native speed comparison is claimed.
