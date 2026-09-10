# EXTSIG unconditional binomial tests

MD Anderson catalog entry 42 compares two binomial proportions. This port covers
all five outcome orderings, one- and two-sided probabilities, mid-p versions,
and the accompanying Fisher and chi-square analyses. It supplies the original
99-point nuisance grid and continuous maximization with numerical bounds.

```python
from mdanderson_stats import extsig

# Five successes among 23 subjects versus 13 among 27: the native worked example.
result = extsig(5, 23, 13, 27)
assert abs(result.two_sided.pvalue - 0.05888313802094) < 1e-8
assert result.two_sided.error_bound < 1e-8

historical = extsig(5, 23, 13, 27, nuisance="native_grid", tie_rule="native")
assert abs(historical.one_sided.pvalue - 0.029659682965697244) < 1e-12
```

Inputs are successes and total size for each group, with positive integer sizes
at most 1,000 each. The default bounded method supports a combined size at most
400; the original-grid mode supports the full 1,000-per-group limit. Selected
probabilities below float64's representable range raise an error.

## Ordering and tails

`ordering` can be:

- `difference`: absolute difference in observed proportions.
- `log_likelihood`: maximized unrestricted minus pooled binomial log likelihood,
  equivalently half the likelihood-ratio deviance.
- `chi_square`: Pearson chi-square without continuity correction.
- `unpooled_z`: absolute difference divided by its unpooled binomial standard error.
- `fisher` (default): smaller conditional Fisher tail, with smaller values more extreme.

For each ordering, outcomes at least as extreme as the observed table define the
two-sided rejection region. The one-sided region additionally retains outcomes
in the direction indicated by the **observed** difference. This is the original
EXTSIG data-directed convention, not a direction specified before observing data.
`direction` records the original groups' orientation. Swapping the input groups
preserves the reported probabilities.

The result contains `one_sided`, `mid_p_one_sided`, `two_sided`, and
`mid_p_two_sided`, each an `ExtsigMaximum`. Mid-p assigns half weight to tied
outcomes and has its **own** nuisance maximization; it is not obtained by simply
halving a p-value. Mid-p does not have the usual exact-test size guarantee.

Conditional `fisher_one_sided` uses the smaller tail, while `fisher_two_sided`
uses probability ordering, reusing the package's Fisher implementation. As in
the source, the one-sided Fisher value for all-success or all-failure data is
set to 0.5. `chi_square`, `chi_square_two_sided`, and `chi_square_one_sided`
provide the original asymptotic analysis; the last is half the two-sided tail.

## Nuisance maximization

Under the null, both groups share probability p. Conditional on t total successes,
the table has a hypergeometric distribution. Summing rejection weights within
each total gives coefficients c[t], and the unconditional rejection probability is

`sum(c[t] * BinomialPMF(t; n1+n2, p))`.

This is a polynomial in Bernstein form. Grouping by total successes replaces
repeated sums over the full two-dimensional outcome space with one-dimensional
probability evaluations. Conditional probabilities use normalized log binomial
coefficients; cumulative Fisher tails are also evaluated in the log domain.
Likelihood-ratio ordering uses direct KL expressions to avoid subtracting large
fitted log likelihoods.

With `nuisance="bounded"`, the algorithm subdivides the complete interval [0,1].
The largest Bernstein coefficient bounds the polynomial on each interval. It
refines any interval whose upper bound can improve the best attained value,
including regions away from initial grid peaks. Thus it does not assume a
single maximum or rely only on a grid search. Each of the four rejection
probabilities is maximized separately.

`ExtsigMaximum` exposes:

- `pvalue`: the returned numerical upper bound on the continuous maximum.
- `lower_bound`: the best attained value minus floating-point padding.
- `nuisance_probability`: a point attaining that lower-bound calculation.
- `error_bound`: the difference between the reported upper and lower bounds.

`tolerance` defaults to 1e-9 and applies to the normalized coefficient scale;
small probabilities are rescaled before maximization, preserving extreme tails.
The nuisance location itself has no requested accuracy guarantee: a flat maximum
can be well bounded before its location is known precisely. Floating-point
padding accounts for coefficient evaluation and subdivision, but these are
**numerical bounds, not a formal interval-arithmetic certificate**. Failure to
resolve a bound within the subdivision limit raises an error.

`nuisance="native_grid"` instead maximizes over p=0.01,...,0.99, exactly the
original finite grid. `error_bound=0` then means no optimization gap within that
finite grid; it says nothing about the gap to the continuous maximum. This mode
can underestimate the true supremum. For instance, the one-sided rejection
probability for 0/1 versus 2/2 with difference ordering is p²(1−p), whose true
maximum 4/27 occurs at p=2/3, between native grid points.

## Ties and native comparisons

Default `tie_rule="inclusive"` includes values tied within relative tolerance
1e-12 on the stable ordering scale. `tie_rule="native"` uses the original strict
`statistic >= cutoff` membership followed by its approximate tie check for mid-p.
It also uses the original likelihood subtraction and `1 − Fisher-tail` ordering
scales. Consequently the native mode retains cancellation and floating-point tie
sensitivities, especially when Fisher tails are extremely small. The default
preserves their log-tail distinctions and includes numerically equal outcomes.
Zero unpooled variance with unequal proportions is infinitely extreme in the
default; the native rule uses the original sentinel 999.

The original Fortran `sigval` was compiled from the 1.4 archive, preserving all
statistical routines. Its internal procedures were placed in a module and their
file-I/O dependencies linked only to make them callable without the interactive
program. All five orderings and all four native-grid outputs for (5,23,13,27)
agree with Python to relative tolerance 2e-12. For the default Fisher ordering,
the original one-sided value is 0.029659682965697244 and its two-sided value is
0.05888313802093932. The manual's displayed Fisher and chi-square probabilities
also agree.

Focused tests check these 20 native outputs, continuous bounds against analytic
maxima for 0/1 versus 2/2 under all orderings, symmetry, degenerate data, and the
extreme two-sided probability 2^-399 for completely separated groups of size 200.

Sources: [MD Anderson EXTSIG](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/42)
and [EXTSIG 1.4 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/EXTSIG/EXTSIG_V1.4.zip).
The bundled documentation still labels itself version 1.3. Archive and source
hashes are recorded in `extsig-sources.json`; original files are not redistributed.
