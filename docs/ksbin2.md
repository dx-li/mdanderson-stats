# KSBIN2 two-sample binomial designs

Catalog entry 24 is partial. The outcome statistics and complete single-stage
outcome ordering are implemented; probability calculations, multistage transitions,
rejection/quitting selection, power-loss tables, probability scans and report/design
workflows remain pending. Source: KSBIN2_V1.tar.gz, ksbin290_2.1.

```python
from mdanderson_stats import ksbin2_ordering, ksbin2_statistic

space = ksbin2_ordering(20, 15, criteria=(1, 2), alternative="greater")
print(space.events, space.score, space.group_end)
score = ksbin2_statistic([1, 2, 3], 10, 1, 10, criteria=(3,), alternative="greater")
```

`ksbin2_statistic` broadcasts both event counts and sample sizes. Each group has
1–100 observations and an integer event count between zero and its sample size.
`greater` means p1 > p2, `less` means p1 < p2, and `two-sided` uses either direction.
Smaller scores indicate stronger evidence for the specified alternative.

Criteria are distinct integers, supplied in the desired order:

| Criterion | Magnitude before direction adjustment |
| --- | --- |
| 1 | Absolute observed difference in proportions |
| 2 | Maximized separate-probability log likelihood minus maximized common-probability log likelihood; not twice that difference |
| 3 | Uncorrected Pearson chi-square for the 2-by-2 table |
| 4 | Absolute difference divided by the unpooled estimated standard error |

For zero variance, criterion 4 is zero for equal proportions and 999 for unequal
proportions, preserving the source's finite sentinel. All magnitudes are negated;
a one-sided outcome pointing against the specified alternative reverses that sign.
Multiple criteria combine as a weighted sum with weights 1, 0.001, 0.000001, etc.
This reproduces SSSSRT; it is not strict lexicographic tie-breaking. A sufficiently
large secondary criterion can change the primary ordering.

The source manual has two inconsistencies: its README calls this a single-binomial
program, and its prose describes criterion 4 as Fisher's exact test. The main
manual introduction, executable menu, and numerical source establish that it is a
two-sample program and criterion 4 is the unpooled Z statistic.

`ksbin2_ordering` enumerates every count pair for scalar group sizes, sorts by score,
and exposes zero-based inclusive ending row indices for the tied groups. Ties use
the source's adjacent-score rule: absolute difference divided by the larger of
absolute-score sum and 1e-100 must be less than 1e-12. Within ties, Python preserves
row-major input order; the source quicksort can permute tied rows. Mathematically
equal proportions get an exact zero log-likelihood ratio, correcting source
roundoff that can otherwise spuriously separate equal-evidence outcomes.

Validation: `tools/reference_ksbin2.py` extracts unchanged SSSSRT (including CHISQ),
XLR (including XMLBIN), and QEQDBL into a local binary64 Fortran reference module.
An independent driver evaluates complete grids for sizes (3,4), (10,10), (20,15),
all three directions and six criterion lists. All 54 grids are retained with
source/extracted hashes and compiler provenance in `tests/fixtures/ksbin2.json`.
Python matches at 2e-12 relative / 1e-13 absolute tolerance. Independent identities,
endpoint cases, direction symmetry, weighted combinations, complete-space coverage,
tied groups and invalid inputs are tested in `tests/test_ksbin2.py`.

These scores order evidence; they are not p-values. No significance, power or
multistage design validity is claimed by this component alone.
