# SURVAN descriptive statistics and frequency distributions

```python
import numpy as np
from mdanderson_stats import survan_describe, survan_frequencies

summary = survan_describe([[1, 4], [2, np.nan], [3, 8]])
assert np.allclose(summary.mean, [2, 6])
assert np.allclose(summary.variance, [1, 8])
frequency = survan_frequencies([0, 1, 1, np.nan])
assert np.allclose(frequency.percent, [100 / 3, 200 / 3])
```

`survan_describe(x)` summarizes each column; a vector produces a one-column result.
It returns counts, missing counts, mean, sample variance, standard deviation,
standard error, legacy standard error, minimum, maximum and range. NaN denotes
missing data and is omitted separately for each column; infinity is rejected.
All-missing column summaries are NaN, and sample variance/SD/errors are NaN when
fewer than two observations remain. Results are immutable one-dimensional arrays.

Variance uses denominator n−1. The usual standard error of the mean is
SD/sqrt(n). The original SUMMT routine instead calculates SD/(n−1); that value is
available explicitly as `legacy_standard_error` and matches the manual's report.
It should not be confused with the usual estimated standard error of the mean.

The Python calculation centers around an observed value and scales before
computing second moments. This avoids the source's subtraction of large raw
sums of squares. For example, `1e12 + [0, 1, 2]` has sample variance 1. Source
minimum/maximum initialization to ±65535 and missing-dispersion sentinels are not
retained. Nonrepresentable variance or range raises an error rather than silently
returning infinity or an underflowed zero variance. Limits are 1,000,000 rows,
200 columns and 2,000,000 entries.

`survan_frequencies(x)` returns sorted distinct values, counts, percentages,
cumulative percentages, missing count and nonmissing observation count.
Percentages use the nonmissing denominator. An entirely missing vector returns
empty frequency arrays. Exact float64 equality is the default, implemented with
NumPy's sorted unique/count operation.

`legacy_grouping=True` reproduces the source's first-matching representative rule:

abs(a−b) / max(abs(a)+abs(b), 1e-20) < 1e-5.

Representatives retain the value of their first observation. Matching happens in
insertion order, and output is sorted only after grouping. The rule is not
transitive: starting with 1.000015 merges [1, 1.000015, 1.00003] into one group,
whereas starting with 1 yields two groups. This mode uses the original rule in
float64 rather than reproducing every single-precision boundary decision. The
ratio is evaluated after scaling to avoid overflow. Frequency input is limited
to 1,000,000 values; approximate grouping additionally limits representative
comparisons to 10,000,000 and raises if that work limit is exceeded.

For imported native data, decode missing sentinels into NaN and the legacy
1e-20 zero sentinel into zero before calling. If a survival column encodes
censoring by a negative sign, summarize/frequency-count its absolute times.
These conversions are explicit because arbitrary negative values and 1e-20
are otherwise valid measurements. Apply any desired record selection before
passing the data; the native descriptive routines themselves read all records.

## Validation

Original SUMMT and FREQ were compiled unchanged with a small observation-reader,
missing-value and initialization harness. All manual-example mean/SD/legacy-SE,
minimum/maximum/count outputs agree within relative 2e-6; group frequencies match
exactly. Focused checks also cover the corrected standard error, large-offset
variance, missing columns, nonrepresentable variance and order-dependent grouping.
See [SURVAN coverage](survan-coverage.md) and [source hashes](survan-sources.json).
