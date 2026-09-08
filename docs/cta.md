# CTA contingency-table analysis

Catalog entry 30 is partial. CHISQT expected counts, percentages, Pearson,
Yates and source-specific Cochran statistics, and the McNemar decomposition are
implemented. Fisher probabilities,
Cohen kappa and its variances, sensitivity/specificity,
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


## McNemar and heterogeneity decomposition

```python
from mdanderson_stats import mcnemar_analysis

fit = mcnemar_analysis([[3, 1, 7], [2, 6, 3], [4, 8, 9]])
print(fit.pairs, fit.pair_statistic, fit.pair_pvalue)
print(fit.summed_statistic, fit.summed_pvalue)
print(fit.pooled_statistic, fit.heterogeneity_statistic)
```

`mcnemar_analysis` implements MCNEMAR for square tables, with optional leading
batch axes. Diagonal cells do not enter the calculation. For each i<j, above=b
and below=a give statistic (b-a)²/(a+b), with one degree of freedom. The summed
statistic adds these contributions. The pooled directional statistic is
(sum(b-a))²/sum(a+b), with one degree of freedom. Heterogeneity is the difference
between summed and pooled statistics, with one fewer degree of freedom than the
summed test. Pooled direction and heterogeneity depend on the supplied category
ordering; this is the source decomposition, not a Stuart-Maxwell marginal-
homogeneity test. No continuity correction is used.

To avoid cancellation near homogeneous pair contrasts, heterogeneity is evaluated
as sum(w*(r-rbar)²), where w=a+b, r=(b-a)/w, and rbar=sum(b-a)/sum(w). Raw large
counts are not squared. The result retains zero-based category pairs, above/below
counts, all three aggregate statistics/p-values/dfs and individual statistics.
All result arrays are read-only.

Pairs with no discordant observations contribute zero and are excluded from the
summed degrees of freedom; their individual p-values are NaN. With no discordance,
all statistics and dfs are zero and all p-values are NaN. A 2x2 table has no
heterogeneity degrees of freedom, so that p-value is NaN. The archived routine
instead divides by zero on empty pairs and prints p=1 for zero heterogeneity with
df=0. Python defines those no-information cases explicitly. Nonnegative fractional
counts are supported; nonsquare/negative/nonfinite inputs and overflowing totals
are rejected.

Seven native MCNEMAR runs cover 2x2, 3x3 and 4x4 tables, symmetry, fractional
counts and differing pair contrasts. `tools/reference_cta_mcnemar.py` records them
in `tests/fixtures/cta_mcnemar.json`, using unchanged archived subroutines. Aggregate
statistics agree within float32 tolerances. The source's OVERFL routine always
returns 1, forcing its Wilson–Hilferty approximation in the x>=df integer-df
branch above df=2. In the recorded 3x3 and 4x4 cases this changes the summed
p-value by about 5e-5 and 3.08e-4 respectively. Tests explicitly reproduce that
approximation to explain native output, and independently check the Python
p-values using finite gamma recurrences from exp/erfc. The implementation uses
accurate gamma tails, not the source overflow stub's forced approximation.

Further checks cover rational decomposition, transpose/orientation, zero/one
active pair, batch scaling to enormous counts and stable positive heterogeneity
for nearly homogeneous contrasts. Fisher, kappa and the other pending CTA
workflows are still tracked above; this does not complete catalog entry 30.
