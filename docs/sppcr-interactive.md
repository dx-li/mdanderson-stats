# SPPCR interactive experiment entry

`read_sppcr_interactive` reads one experiment from text streams and returns the
same validated `SPPCRData` used by batch and FileMaker input. The complete SPPCR
menu and report/file workflow remain outstanding.

```python
import numpy as np
from mdanderson_stats import (
    read_sppcr_interactive,
    sppcr_bootstrap,
    sppcr_bootstrap_intervals,
)

# Defaults to stdin/stdout; supplied streams stay owned by the caller.
data = read_sppcr_interactive()
bootstrap = sppcr_bootstrap(
    data.dna,
    data.seen,
    data.wells,
    progenitor=data.progenitor,
    rng=np.random.default_rng(8),
    replicates=1000,
)
limits = sppcr_bootstrap_intervals(bootstrap)
```

## Prompts, bounds and units

The field order and bounds follow `problem_in_mod.interactive_read`:

1. number of runs, 1–50;
2. number of alleles, 1–50;
3. one distinct integer size per allele, each 1–999;
4. two progenitor sizes appearing in that list, repeated for a homozygote;
5. one genome-equivalent DNA amount per run, each .001–1000;
6. one integer well count per run, each 1–500;
7. one integer detection count per allele for each run, each 0–that run's wells.

All bounds are inclusive. Allele sizes are labels, not numeric model parameters.
The Python prompt prints the full allele order for each run, rather than only the
source's first 20 labels. DNA conversion is performed once by `sppcr_data`: file
and entered units are retained in `genome_dna`, while `dna` contains twice those
values for the numerical APIs.

Input uses the existing tested `CDFConsole` numeric reader. Lists may continue
onto additional lines. Integer syntax, decimal/D exponents, comma separators and
repeat notation follow its documented contract. Excess fields on the final
record of a numeric request are discarded, matching native list-directed input.
A malformed or out-of-range vector restarts the entire requested vector; accepted
prefixes from a failed attempt are not silently reused.

Duplicate allele-size lists and progenitor pairs absent from the size list can
be corrected. The source does not validate uniqueness and can print that it is
aborting after an absent progenitor without actually stopping; Python uses bounded
corrections and never returns an invalid mapped progenitor index.

## Failure and ownership contracts

`max_attempts` defaults to 3. It bounds retries for each numeric request and the
separate identity checks. A numeric retry sequence can occur within an identity
retry, so these are not a single global attempt counter. `max_records` defaults
to 10,000 and applies per numeric request, across that request's numeric retries.
`max_line_length` defaults to 10,000 input characters. Exhausting these limits
raises `CDFConsoleError`. Invalid configuration is rejected before reading input.

EOF raises EOFError and returns no partially initialized experiment. Read/write
failures propagate. Supplied streams are neither closed nor replaced. Independent
calls can read successive experiments from the same stream without shared data
arrays. When streams are omitted, stdin/stdout are used.

`unseen_alleles="drop"` follows native conversion for never-seen nonprogenitor
alleles and prints the omitted labels. Dropping an unseen progenitor raises the
shared data-validation error after counts are entered. Use
`unseen_alleles="retain"` up front to keep all labels, including an unobserved
progenitor or all-zero experiment. This final data-policy error does not trigger
an implicit restart of the whole dialogue. All returned arrays are immutable.

The function collects data only: it does not run a bootstrap, reset random seeds,
open report files or launch the full application menu. The explicit composition
above makes those subsequent operations reviewable and reproducible.

## Validation

`tools/reference_sppcr_interactive.py` compiles and calls the original interactive
reader with controlled unused input storage. The fixture retains its driver,
archive hash, supplied transcripts, return codes and native output. Tests compare
resulting DNA, counts, well counts, sizes and mapped progenitors for ordinary
entry, continued vectors, rejected numeric input, excess final-record fields and
omitted unseen alleles. Exact prompt typography is not a compatibility target.

Additional tests cover duplicate-label and progenitor corrections, out-of-range
vectors, explicit unseen-progenitor policy, immutable independent results,
successive experiments, EOF at multiple stages, retry and input-size limits,
configuration validation, output failure propagation, and entry through bootstrap
fits and confidence intervals. Console parsing is reused; no performance speedup
is claimed for human-paced data entry.
