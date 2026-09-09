# STATTAB distribution results

`stattab_solve` implements the numerical result layer for all twelve STATTAB
families and all 42 supported computed groups. It connects the independently
validated CDFLIB kernels to the application's parameter order, output columns,
extra probabilities and neighboring integer rows. The interactive parser, list
editor integration, session reuse, formatted reporting and complete reconciliation
of the shared source versions remain unfinished.

```python
from mdanderson_stats import STATTAB_DISTRIBUTIONS, stattab_solve

normal = stattab_solve("normal", x=[-1, 0, 1], mean=0, sd=1)
print(normal.columns)  # x, mean, sd, cum, ccum, two_sided_p
print(normal.values)  # one row per input

# Source gamma inputs A and B are named rate and shape explicitly.
gamma = stattab_solve("gamma", x=1, rate=2, shape=3)
assert gamma.columns == ("x", "shape", "rate", "cum", "ccum")

# Compute either member of a complementary group, supplying the smaller tail.
quantile = stattab_solve("normal", compute="x", ccum=1e-20, mean=0, sd=1)

counts = stattab_solve("poisson", compute="s", mean=3, cum=0.5)
print(counts.parameters["s"])  # continuous inverse, not an integer PPF
for neighbor in counts.neighbors:
    print(neighbor.kind, neighbor.valid, neighbor.values)
```

## Parameter contract

The distribution names and input order below follow the source menu, with clear
Python parameter names. Every family also has `cum` and `ccum` as its last input
parameters. `compute` defaults to `cum`; either name in a complementary group
selects the same calculation. `STATTAB_DISTRIBUTIONS` is an immutable mapping of
menu numbers, input names and computed groups. Group order follows the numerical
kernel selectors; gamma's shape/rate group order differs from its input order.

| Distribution | Non-tail input parameters | Other computed groups |
|---|---|---|
| `beta` | `x`, `cx`, `a`, `b` | `x/cx`, `a`, `b` |
| `binomial` | `s`, `n`, `pr`, `cpr` | `s`, `n`, `pr/cpr` |
| `neg_binomial` | `f`, `s`, `pr`, `cpr` | `f`, `s`, `pr/cpr` |
| `chisq` | `x`, `df` | `x`, `df` |
| `nc_chisq` | `x`, `df`, `pnonc` | `x`, `df`, `pnonc` |
| `f` | `f`, `dfn`, `dfd` | `f` |
| `nc_f` | `f`, `dfn`, `dfd`, `pnonc` | `f`, `pnonc` |
| `gamma` | `x`, `rate`, `shape` | `x`, `shape`, `rate` |
| `normal` | `x`, `mean`, `sd` | `x`, `mean`, `sd` |
| `poisson` | `s`, `mean` | `s`, `mean` |
| `t` | `t`, `df` | `t`, `df` |
| `nc_t` | `t`, `df`, `pnonc` | `t`, `df`, `pnonc` |

Omit the computed group. Supply every other parameter explicitly, except one
member of a complementary pair may be omitted. Passing `None` is an error;
normal mean/SD and gamma rate do not silently default in this application API.
Both supplied members must satisfy the existing CDFLIB pair contract, which
preserves the smaller member and reconstructs its complement. Unknown names,
missing parameters and invalid domains raise `ValueError`. Numerical failures
propagate explicitly; no failed or nonfinite result is returned as a valid table.

All inputs broadcast, including parameters other than the first coordinate and
complementary values. Empty batches are supported. The numerical domains and
inverse limitations are those documented for the package's corresponding
[`cdf_*` APIs](cdflib90-coverage.md), including their deliberate repairs. F and
noncentral F degree-of-freedom inversions are excluded, as in STATTAB. For
noncentral-t DF inversions, `df_bracket=(low, high)` selects a sign-changing
interval; endpoints may broadcast. Multiple roots are not enumerated, and the
default full-domain search does not promise to find every possible root.

Poisson forward evaluation additionally accepts exactly zero mean, with lower
tail one and upper tail zero at every nonnegative count. Positive means retain
the CDFLIB90 [1e-10,1e100] domain and counts [0,1e100]. The zero-mean subset uses
the existing validated legacy Poisson kernel. Inversions retain the positive-mean
CDFLIB90 contract; zero-mean count inversion is not identifiable. Standalone
[discrete terms](stattab-probability.md) have a wider finite-input domain.

## Results and neighboring rows

`STATTABResult.parameters` holds completed parameter arrays in source input
order. `columns` describes the final axis of `values`; its preceding axes are
the input broadcast shape. A scalar call has a one-dimensional output row.
The arrays and mapping are immutable and independent of caller input storage.
Forward gamma columns put shape before rate; inverses put the probability pair
first, then fixed parameters, then the computed group, matching the source.

Only forward binomial, negative-binomial and Poisson results append `term`.
Counts are truncated for that discrete mass while the CDF retains its continuous
extension. Only forward normal and Student-t results append `two_sided_p`, equal
to twice the smaller tail. Chi-square and F upper tails already serve the source's
many-sided probability column; no additional density-ordered test is introduced.
Extra calculations do not overwrite the saved CDF or any other parameter.

The five count inversions—both counts for binomial and negative-binomial, and
the Poisson count—retain the continuous answer and two `STATTABNeighbor` objects:
`floor` and `floor_plus_one`. As in the source, an integral solution still has
both candidates. The other parameters are not truncated for these CDF rows.
Neighbor columns are identical to the main inverse result's columns.

Each neighbor carries:

- `candidate`: the candidate count in the full broadcast shape;
- `valid`: an immutable boolean mask in that shape;
- `source_indices`: flattened batch indices of valid candidates;
- `values`: compact rows for those indices, shaped `(number_valid, number_columns)`.

Candidates outside the kernel's count bounds, binomial successes above trials,
or trials below successes are invalid. `floor_plus_one` is also invalid when its
increment cannot be represented exactly in float64. Its stored `candidate` may
then be rounded; it must not be used without checking `valid`. Invalid candidates
have no numerical row, so no NaN, stale answer or fabricated tail is needed.
The continuous main answer remains available even if a neighboring row is invalid.
A numerical failure evaluating an otherwise valid neighbor raises an error.

## Validation and performance

Tests exercise every computed group with independent targets: exact discrete
sums, beta polynomials, exponential/integer-gamma identities, normal erfc,
a closed-form Student-t integral, Poisson mixtures and conditional noncentral-t
quadrature. Source column sequences are checked explicitly. Native forward rows
are compared at their printed precision where valid; the failed native noncentral
chi-square tails and negative-binomial mass are rejected as numerical oracles.

Further checks cover every forward input position, both tail-input positions for
inverse tables, mixed-axis broadcasting, empty batches, all five neighboring-row
workflows, invalid and unrepresentable neighbors, tiny two-sided probabilities,
noncentral-t multiple-root brackets, immutable ownership and input failures.
The existing kernel and STATTAB probability tests remain part of validation.

[Benchmarks](stattab-results-benchmark.json) compare batched evaluation with repeated
calls to this same Python API, checking equal main output rows and reporting three-run
medians. They measure batching overhead, not a speedup relative to native Fortran.
There is no per-element Python numerical loop in the result layer; loops assemble
a fixed number of columns, groups or neighboring rows.
