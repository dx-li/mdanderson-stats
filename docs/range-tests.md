# RANGE2 and KWRANGE

`range2` and `kwrange` port the numerical and grouping functionality of catalog
entries 53 and 48. Both accept vectors of group summaries, sample sizes, and a
critical value, and return all pair statistics, rejection decisions, sort order,
and similarity groups. Returned matrices use the original input order and group
indices are zero-based. Group labels can be attached by the caller using those
indices. No significance level or degrees of freedom are inferred from a cutoff.

```python
from mdanderson_stats import range2, kwrange

comparison = range2([9, 1, 4], [3, 20, 5], error_mean_square=2, critical_value=8)
historical = range2([9, 1, 4], [3, 20, 5], 2, 8, legacy=True)
ranks = kwrange([119, 6, 85], [7, 3, 10], 3.5, rank_sums=True)
```

## Statistics and critical values

For RANGE2, the pair statistic is

\[
q_{ij}=\frac{|\bar x_i-\bar x_j|}
 {\sqrt{\mathrm{MSE}(1/n_i+1/n_j)/2}}.
\]

Supply the Tukey studentized-range cutoff for the desired number of groups,
residual degrees of freedom, and significance level. The original executable
asks for degrees of freedom and a p-value label but never uses those two inputs
after the cutoff is supplied; they are omitted from the Python calculation.

For KWRANGE, rank sums are divided by sample size when `rank_sums=True`.
The archive computes the statistic

\[
t_{ij}=\frac{|\bar R_i-\bar R_j|}
 {\sqrt{N(N+1)(1/n_i+1/n_j)/12}},\quad N=\sum_i n_i.
\]

The Python port retains this denominator. Although the original interface asks
for a Tukey critical value, this statistic differs by a factor of sqrt(2) from
the conventional Nemenyi studentized-range statistic. To apply that calibration,
compare `t` with the studentized-range cutoff divided by sqrt(2). The
[PMCMRplus Nemenyi documentation](https://github.com/cran/PMCMRplus/blob/master/man/kwAllPairsNemenyiTest.Rd)
explicitly applies the sqrt(2) transformation before evaluating the studentized
range distribution. This port does not apply a tie correction or silently change
the user-supplied cutoff.

## Original behavior and explicit compatibility

The default keeps each sample size associated with its mean, evaluates all pairs,
and reports maximal contiguous groups in sorted-mean order for which every pair
is nonsignificant. Contiguous groups summarize the matrix; noncontiguous
nonsignificant pairs may also exist and are retained in `reject`.

Setting `legacy=True` reproduces two behaviors in both originals:

1. The program sorts means and group labels without sorting the sizes. Unequal
   sizes can therefore become attached to the wrong means. For means `[9,1,4]`,
   sizes `[3,20,5]`, MSE=2 and cutoff=8, this changes the first-versus-third-group
   decision. Equal-size groups are unaffected by this particular defect.
2. Once the endpoints of a sorted range are nonsignificant, the program marks
   every internal comparison nonsignificant, without evaluating each pair. With
   heterogeneous sizes, an interior pair can nevertheless exceed the cutoff.
   In compatibility mode `reject` preserves that grouping decision while
   `statistic` still exposes every calculated pair statistic.

Legacy mode also preserves the original sorting algorithm's ordering of tied
means. Uninitialized diagonal characters printed by Fortran are not emulated:
the Python diagonal statistic is zero and rejection is false. A zero MSE is
handled explicitly: equal means have statistic zero, different means infinity.

## Validation and performance design

`tools/reference_ranges.py` compiles and executes the unmodified RANGE2 and
KWRANGE sources. Its fixture records 39 runs, covering equal/unequal sample
sizes, sorted/unsorted and tied means, several cutoffs, and both mean-rank and
rank-sum input. Compatibility decisions, sort order and similarity groups match
the original output exactly in those cases. Tests also verify hand-calculated
pair statistics, permutation invariance of default decisions, edge conditions,
and similarity groups against an exhaustive enumeration on random small inputs.

Pair statistics and decisions use NumPy broadcasting. Default similarity groups
are computed from the last incompatible predecessor of each sorted group, with
quadratic total work and storage, matching the size of the result matrix.
The original interactive programs' 60-group limit is not imposed.

The legacy grouping code is a modified Python adaptation. The original
[notices](../THIRD_PARTY_NOTICES.md) are preserved and included in wheel builds.
