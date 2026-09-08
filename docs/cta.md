# CTA contingency-table analysis

Catalog entry 30 is partial. CHISQT expected counts, percentages, Pearson,
Yates and source-specific Cochran statistics, the McNemar decomposition, and
Cohen kappa with variances, sensitivity/specificity and predictive values are
implemented, along with odds ratios, confidence limits and Fisher probabilities.
Binomial comparison, reports
and interactive study orchestration
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


## Cohen kappa and variances

```python
from mdanderson_stats import cohen_kappa

fit = cohen_kappa([[20, 2], [9, 1]])
print(fit.kappa, fit.variance, fit.null_variance)
source = cohen_kappa([[20, 2], [9, 1]], legacy=True)
```

`cohen_kappa` implements unweighted two-rater agreement for square tables, with
leading batch dimensions and finite nonnegative fractional frequencies allowed.
Proportions, observed/chance agreement, kappa, large-sample variance and variance
under independence are returned as read-only arrays. Input is not mutated.
There are no weighted-kappa, confidence-interval or hypothesis-test additions in
this workflow; the source returns the coefficient and these two variances.

Let p_ij be joint proportions, r_i row margins, c_i column margins,
po=sum(p_ii), and pe=sum(r_i*c_i). Kappa=(po-pe)/(1-pe). The default large-sample
variance uses the multinomial delta method: g_ij=[I(i=j)*(1-pe) -
(1-po)*(c_i+r_j)]/(1-pe)^2, and variance=[sum(p*g²)-sum(p*g)²]/n.
The implementation evaluates the centered squared gradients for numerical
stability. The null variance evaluates the same delta method at independent
joint probabilities r_i*c_j, with po=pe. This calculation is symmetric when
swapping the two raters.

The archived KAPPA code instead computes theta4=sum(p_ij*(r_i+c_j)^2) and
theta5=sum(c_i²*(c_i+r_i)). The corrected terms are
sum(p_ij*(c_i+r_j)^2) and sum(r_i*c_i*(r_i+c_i)), respectively. `legacy=True`
retains the executable's original theta indices and expanded variance formulas.
It does not clip negative legacy values or reinterpret them as standard errors.
For [[20,2],[9,1]], kappa is .01123596 in both modes, while default variances
are .01948056 and .01887782; source variances are .06786340 and -.06065759.
The source null variance can also change when the raters are swapped. These are
explicit source-compatibility outputs, not valid negative variance estimates.

Chance disagreement is summed from off-diagonal independent probabilities rather
than subtracting an almost-one agreement from one. Kappa is computed as
1-observed_disagreement/chance_disagreement. Default variance terms multiply
square-root probabilities before squaring to reduce overflow. If chance
disagreement is zero (both raters always use the same single category), kappa
and both variances are undefined NaN. Perfect agreement across multiple occupied
categories has kappa one and zero large-sample variance. Empty tables, invalid
shapes, negative/nonfinite counts and nonrepresentable totals are rejected.

Nine native KAPPA cases, including transposed unequal margins and a negative
legacy null variance, are captured in `tests/fixtures/cta_kappa.json` by
`tools/reference_cta_kappa.py`. The unchanged archived routine matches legacy
results within relative 1e-5/absolute 1e-8 tolerances for its float32 arithmetic.
Independent Fraction-based derivatives and multinomial covariance calculations
check default coefficients and both variances, including independent null tables.
Other tests cover transpose symmetry, sample-size scaling, batches, perfect and
undefined agreement, input immutability and extremely rare agreeing categories.
The original routine writes marginal totals into its input workspace; Python
returns new proportions and leaves caller data intact.


## Diagnostic accuracy and predictive values

```python
from mdanderson_stats import diagnostic_accuracy

fit = diagnostic_accuracy([[80, 10], [20, 90]], standard="columns", positive_index=0)
print(fit.sensitivity, fit.specificity)
print(fit.positive_predictive_value, fit.negative_predictive_value)
print(fit.standard_errors, fit.denominators)
```

`diagnostic_accuracy` implements SENSPEC for final 2x2 table axes, with optional
leading batch dimensions. `standard="columns"` means columns are the reference
classification and rows are the test; `standard="rows"` reverses these roles.
`positive_index` is zero or one and identifies the positive class on both axes.
These explicit options replace the source's two prompts; Python indices are
zero-based. Frequencies may be fractional, as in the source.

The four probabilities are sensitivity TP/(TP+FN), specificity TN/(TN+FP),
positive predictive value TP/(TP+FP), and negative predictive value TN/(TN+FN).
`standard_errors` and `denominators` have a final axis in that same order.
Default standard errors are conditional binomial probability errors
sqrt(p*(1-p)/n). They describe each conditional proportion, not uncertainty from
estimated prevalence or other sampling designs.

`legacy=True` preserves the source's sqrt(n*p*(1-p)), which is a count standard
deviation, despite being labeled “Standard Error” beside probabilities in its
report. Thus the legacy output is n times the default probability standard error.
It is not used as the default. Complementary probabilities are evaluated from the
opposite cells rather than 1-p, preserving rare errors when p rounds to one;
square-root factors also avoid unnecessary variance-product overflow. Legacy
mode preserves the mathematical formula, not float32 rounding artifacts.

A zero conditioning margin gives NaN for only that probability and error. With
all counts zero, all four probabilities/errors are NaN and denominators are zero.
Nonnegative finite data are required and overflowing margins are rejected.
Results are read-only and caller arrays are not mutated. Changing the designated
positive class swaps sensitivity/specificity and PPV/NPV; transposing the table
while swapping the standard axis leaves the results unchanged.

Sixteen unchanged native SENSPEC runs cover both standard axes and both positive
indices across ordinary, fractional and perfect-classification tables. Fixtures
in `tests/fixtures/cta_diagnostic.json` are reproducible with
`tools/reference_cta_diagnostic.py`. Sensitivity/specificity and their source
errors are captured from routine outputs; predictive values/errors are read from
the original five-decimal report. Tests allow 5.1e-6 absolute difference for this
report precision. Independent checks cover conditional binomial formulas,
transpose/class reversal, batch sample-size scaling, local undefined margins,
rare-error precision, immutability and invalid arguments. Remaining CTA workflows
are listed at the top of this document.

## Odds ratios and confidence limits

```python
from mdanderson_stats import odds_ratio

fit = odds_ratio([[80, 10], [20, 90]], risk_factor="columns", response_index=0)
print(fit.odds_ratio, fit.lower, fit.upper)
source = odds_ratio([[80, 10], [20, 90]], legacy=True)
```

`odds_ratio` implements RELRISK. Despite the source report's “relative risk”
label, this is an **odds ratio**, not a ratio of response probabilities.
`risk_factor` selects the grouping axis, and `response_index` (zero or one)
selects the response on the other axis. The comparison is always the response
odds in risk group zero divided by those in risk group one. Transposing the
table and changing the risk-factor axis preserves the result. Reversing the
response or exchanging risk groups reciprocates the ratio and interval.

Final input axes must be 2x2, with strictly positive finite cells; fractional
counts are allowed. Leading dimensions form a batch. Zero cells are rejected,
as in RELRISK, with no implicit continuity correction. The returned arrays are
read-only. `odds` contains the two group odds, and `standard_error` is the
standard error of the **log** odds ratio: sqrt(sum(1/cell)).

The default confidence limits are exp(log(OR) ± z * SE), where
z = Phi-inverse(1-alpha/2). `alpha` is a scalar strictly between zero and one,
with default 0.05. These are asymptotic log-Wald intervals, not exact intervals.
The source incorrectly calls its normal **CDF** PHI(alpha/2) instead of an
inverse CDF. `legacy=True` explicitly retains that formula, using an accurate
double-precision CDF rather than reproducing every single-precision rounding.
At alpha=0.05 its multiplier is about 0.50997 rather than 1.95996, producing
much narrower limits. Legacy limits are provided for reproduction only.

Logarithms avoid intermediate odds overflow. Reciprocal square roots and a
scaled Euclidean norm keep log standard errors finite even for subnormal cells.
The default quantile is evaluated from log(alpha)-log(2), retaining even the
smallest positive floating-point alpha. Exponentiation can still yield zero or
infinity when a result exceeds floating-point range; `log_odds_ratio`,
`log_lower`, and `log_upper` retain the corresponding finite log values.

`tools/reference_cta_odds.py` compiles unchanged RELRISK locally and records 48
native cases across four tables, both risk-factor axes, both response indices,
and three alpha values. The driver initializes the success flag because RELRISK
only sets it on failure. `tests/test_cta_odds.py` compares all four returned
native quantities, independently checks corrected intervals using standard
normal quantiles, and covers batching, reversal, scaling, validation and
extreme floating-point inputs. Original Fortran and executables remain outside
the package. These checks do not validate CTA's remaining routines.

## Fisher fixed-margin probabilities

```python
from mdanderson_stats import fisher_exact

fit = fisher_exact([[1, 9], [11, 3]])  # Probability-ordered two-sided test
lower = fisher_exact([[1, 9], [11, 3]], alternative="less")
source = fisher_exact([[1, 9], [11, 3]], legacy=True)
full_source_tail = fisher_exact([[1, 9], [11, 3]], alternative="source")
```

`fisher_exact` implements FISHXT and standard exact-test alternatives. Final
axes must be 2x2 with nonnegative integer counts. Leading dimensions form a
batch; each table may total at most 50,000, the original CTA workspace limit.
Zero margins, including an all-zero table, yield the degenerate probability
one. Fractional counts are rejected: the original implicitly truncates them
inside factorial loops, which does not define a valid fixed-margin test.

For a table [[a,b],[c,d]], the upper-left count has a hypergeometric distribution
with population a+b+c+d, first-row size a+b, and first-column size a+c. The
observed table is included in every tail:

- The default `two-sided` sums all table probabilities no greater than the
  observed probability; equal masses are included with relative tolerance 1e-12.
- `less` sums upper-left counts at most a; `greater` sums counts at least a.
- `source` selects `less` when a*d < b*c and `greater` otherwise, including
  the equality case. This is CTA's direction chosen from the data, not a
  pre-specified one-sided alternative or a two-sided p-value.

`legacy=True` defaults to `source` and cannot be combined with another explicit
alternative. It also reproduces CTA's early stop: start at the observed table,
move outward in the selected tail, and stop **after including** the first term
less than 1e-5 times the observed table's probability. Without legacy, the full
selected tail is summed. For [[100,100],[100,100]], legacy includes 25 terms;
the full upper tail includes 101. Standard two-sided inference does not use
this truncation.

Results contain `pvalue`, `observed_probability`, the number of included `terms`,
`support_size`, a Boolean `source_lower_tail` describing CTA's direction, and
`truncated` indicating whether the early stop actually omitted any terms.
Arrays are read-only. Supports are evaluated using vectorized hypergeometric
probabilities, one table at a time to bound batch memory use. Relative cutoffs
use log probabilities so they remain meaningful even when the probability
itself underflows to zero. Two-sided ordering of underflowed masses uses log
probabilities with an absolute log tolerance of 1e-10. Probabilities below the
floating-point range return zero.

`tools/reference_cta_fisher.py` compiles unchanged FISHXT and ROTCAF locally,
using the original 50,000-element work buffer. Its 40 fixtures exercise table
transposition, row reversal, zero margins, independence and tail truncation.
All source term counts match. Native probability comparisons allow 7e-4 relative
error: CTA's repeated single-precision log-factorial sums differ from the exact
rational result by about 6.28e-4 at total 400. Python does not reproduce that
roundoff. Every native case is also checked against rational sums to 2e-13
relative tolerance. Further tests exhaustively enumerate all 625 tables whose
cells range from zero through four for all four alternatives, using integer
combinations and fractions as an independent oracle. The 50,000-count boundary,
underflow, batching, symmetry and invalid inputs are covered separately.
