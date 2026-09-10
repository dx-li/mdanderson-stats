# SPPCR analysis and replicate reports

`format_sppcr_report(data, bootstrap)` returns a complete text analysis for one
experiment. `format_sppcr_simulations(data, bootstrap)` returns a table of every
replicate estimate. These formatters consume validated results; they do not refit,
resample, open files or mutate random state. [Truth entry and reports](sppcr-truth.md)
and [composed analysis workflows](sppcr-analysis.md) are available, with modern or
[historical RNG support](sppcr-random.md). The full menu/file workflow remains pending.

```python
import numpy as np
from mdanderson_stats import (
    sppcr_data,
    sppcr_bootstrap,
    format_sppcr_report,
    format_sppcr_simulations,
)

data = sppcr_data(
    [0.5, 1],
    [[4, 8, 2], [10, 15, 5]],
    [20, 30],
    [100, 102, 104],
    (100, 102),
)
bootstrap = sppcr_bootstrap(
    data.dna,
    data.seen,
    data.wells,
    progenitor=data.progenitor,
    rng=np.random.default_rng(92),
    replicates=1000,
)
analysis_text = format_sppcr_report(data, bootstrap)
replicate_tsv = format_sppcr_simulations(data, bootstrap)
```

## Analysis content

The report uses labeled sections with tab-separated headers and rows:

- metadata: interval multiplier, replicate count, undefined frequency replicate
  count, progenitor sizes and omitted never-seen allele labels;
- `DATA`: original detection counts, well counts, input genome-equivalent DNA and
  model DNA (twice the input amount);
- `SAMPLING PROBABILITIES`: the actual detection probabilities used to generate
  replicates, including any caller-supplied model;
- `MEANS`: observed mean estimates, bootstrap means, asymptotic and bootstrap SDs,
  solver brackets, observed adjustment/boundary flags and bootstrap counts of
  adjusted or boundary fits;
- `SUMMARY`: calibration, inverse calibration and combined mutant frequency,
  with confidence bounds, availability and clipping flags;
- `FREQUENCIES`: allele labels, progenitor/mutant roles, observed and bootstrap
  estimates, asymptotic and bootstrap SDs, confidence limits and diagnostics;
- `TRANSFORMED FREQUENCIES`: observed and bootstrap `2*asin(sqrt(p))` values and
  their SDs, for each allele and the combined mutant group.

`P` identifies either progenitor index; `M` identifies other alleles. Boolean
flags use 0/1. Calibration and mean estimates are expressed relative to `DNA_model`.
Both DNA inputs are printed explicitly because the native data report labels do
not clearly expose its factor-of-two input conversion. Original observed counts
are never replaced by the fit's half-count-adjusted data in the data table;
adjustments are separately disclosed.

The default `multiplier=1.959964` matches the source's rounded normal critical
value for nominal 95% bootstrap intervals. Custom positive finite multipliers are
passed to the validated interval API and printed without asserting a confidence
level. These are approximate normal/bootstrap limits, not exact intervals.

`NA` means unavailable or not computed. Inverse-calibration bootstrap mean and
SD columns are intentionally NA: the native report provides its reciprocal point
estimate and inverted interval, not a bootstrap distribution of reciprocals.
`inf` represents an unbounded reciprocal-calibration upper limit when calibration
can reach zero. Other unavailable bounds remain NA with `available=0`. An all-zero
experiment still reports zero calibration and all replicate records, with undefined
frequency estimates and their uncertainty clearly marked. Asymptotic uncertainty
can be NA while empirical bootstrap uncertainty remains available.

## Replicate content

The simulation output is one TSV row per replicate with:

- one-based replicate number and frequency-defined flag;
- calibration, every allele mean and every allele frequency;
- combined mutant frequency;
- number of adjusted alleles and number of boundary alleles.

Column names carry the supplied allele labels. Undefined frequency values are NA;
no replicate is discarded. Generated counts and detailed per-fit diagnostics
remain available in the original bootstrap object. This replaces native `Cal`,
`Mu`, `Freq` text blocks with a labeled table that preserves those values and adds
identity and boundary information.

## Validation, formatting and limits

Reports require one observed experiment. Data counts, well counts, model DNA and
progenitor mapping must match the observed fit and bootstrap summary; mismatches
raise ValueError. The data object's units and identity mapping are revalidated.
Caller-created result containers are expected to obey the numerical APIs' contracts;
the formatter is not a general repair mechanism for manually corrupted results.

`precision` defaults to 10 significant digits and accepts integers 1–17. General
numeric formatting avoids the native fixed-width asterisk overflow. Use 17 digits
when round-tripping printed finite float64 values. Rounded printed solver brackets
may appear equal; the result object retains their full precision. The source's
initial solver guess has no corresponding role in the Python bisection algorithm,
so meaningful solver brackets are reported instead.

`max_characters` defaults to one million for an analysis and ten million for
simulations. Output is bounded while rows are constructed, before a string is
returned. Exceeding the bound raises ValueError and returns no partial report.
Filesystem or stream writing is left to the caller; safe application file handling
will be integrated separately.

Tests parse the rendered tables and check values against the previously validated
fitting, frequency, bootstrap and interval results. They cover every section,
units, identities, all-zero experiments, boundary adjustments, unbounded limits,
all replicate rows, mismatched data, precision/output limits and deterministic
formatting. The report content was also manually inspected. Native prompt and
fixed-width typography are not compatibility targets; the archived report routines
establish the statistical quantities to expose. No speedup claim is made for text
formatting.
