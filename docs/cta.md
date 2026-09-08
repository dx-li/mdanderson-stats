# CTA contingency-table analysis

Catalog entry 30 is partial. CHISQT expected counts, percentages, Pearson,
Yates and source-specific Cochran statistics are implemented. Fisher probabilities,
McNemar/extended McNemar, Cohen kappa and its variances, sensitivity/specificity,
relative odds, binomial comparison, reports and interactive study orchestration
remain pending.

The [official catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/30)
lists version 1, modified March 19, 1992; the downloadable CTA_V1.tar.gz contains
cta0298.f dated February 2, 1998, HOWTOGET and LEGALITIES. Source SHA-256:
`7c25440a91d37c2cb437c486965c1142debcba6e0cf589065504d989c0251834`.
Original source is not bundled. Its legal notice is retained in
[notices/mdanderson-cta-LEGALITIES.txt](../notices/mdanderson-cta-LEGALITIES.txt).
This is an independent Python implementation, not an endorsed original release.

```python
from mdanderson_stats import contingency_chi_square

fit = contingency_chi_square([[12, 5], [7, 16]])
print(fit.expected, fit.statistic, fit.pvalue)
print(fit.row_percent, fit.column_percent)
print(fit.yates_statistic, fit.cochran_statistic)
source = contingency_chi_square([[10, 10], [10, 10]], legacy=True)
```

The final two input axes are rows and columns; leading dimensions form a batch.
Finite nonnegative fractional counts are allowed, as in CHISQT. There must be at
least two rows and columns, and every marginal total must be positive. Empty
margins are rejected rather than silently dropped or converted into NaN results.
The result retains copied observations, expected frequencies, cellwise Pearson
contributions, row/column percentages, degrees of freedom and tail probabilities.
Arrays are read-only. Expected counts use row proportions times column totals;
standardized residuals avoid squaring raw counts or multiplying four large margins.

Pearson uses sum((O-E)^2/E), df=(rows-1)*(columns-1). Default Yates is available
only for 2x2 tables and uses max(abs(O-E)-.5,0)^2/E. `legacy=True` reproduces the
source's unclipped subtraction and applies it to every table shape. Thus an
exactly independent table of four 10s has default Yates statistic zero but source
statistic .1. This source behavior is explicit rather than used as the default.

For 2x2 tables the source-specific Cochran calculation locates the first minimum
expected cell in row-major order and sets d=abs(O-E) there. Its adjusted d is
floor(2*d)/2 if O<=2*E, otherwise d-.5. The statistic is adjusted_d²*sum(1/E),
algebraically equal to the source's total³ times adjusted_d² divided by all four
margins. Half-unit truncation is evaluated without overflowing 2*d. These fields
are None for other shapes. This is the archived correction, not Cochran's Q test.

P-values use the upper regularized gamma function directly, avoiding the source's
subtraction of an approximate chi-square CDF from one. Legacy mode selects source
statistic conventions, not single-precision rounding or its approximate tail
routine. Source truncation near half-unit boundaries may differ when float32
rounding changes which side of a boundary a cell lies on.

`minimum_expected` and `percent_small_expected` retain small-cell diagnostics;
the latter counts E<=expected_threshold (default 5), matching the executable
comparison even though the printed source label says “less than”. These are
returned metadata, not automatic cell merging or a switch to an unfinished
Fisher workflow. Invalid inputs and unrepresentable totals/expected counts fail
explicitly.

## Validation

`tools/reference_cta.py` compiles all archived subroutines unchanged with gfortran,
replacing only the top-level interactive program with a small driver. It invokes
CHISQT with its own CHI2 dependencies and captures expected counts, three
statistics and probabilities. The resulting nine cases in
`tests/fixtures/cta_chisqt.json` cover 2x2, 2x3, 3x3, zero cells, independence,
fractional observations and unequal margins. Nonapplicable native Cochran outputs
are driver sentinels and are represented as None in Python. Compiler flags and
source hash are recorded. Tests use relative tolerance 2e-6 for single-precision
statistics/expected counts and absolute 5e-7 for the native approximate p-values.

Independent tests use rational Pearson calculations, the exact df=2 tail formula,
corrected independence, batch/scalar and transpose/permutation equality, threshold
equality, immutable input snapshots, and counts large enough to overflow a naive
squared-count implementation. Other CTA routines are compiled as dependencies
but are not claimed to have been exercised or ported in this increment.
