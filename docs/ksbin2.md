# KSBIN2 two-sample binomial designs

Catalog entry 24 is partial. The outcome statistics and complete single-stage
outcome ordering and ordinary single-stage probability tables are implemented.
Source-convention mid-p significance and single-stage rejection-region selection
and fixed multistage operating characteristics are also available. Ordinary
multistage boundary-assistance and power-loss tables are implemented. Multistage
mid-p reporting, study probability scans and report/design workflows remain pending. Source: KSBIN2_V1.tar.gz, ksbin290_2.1.

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

These scores order evidence; they are not p-values. The ordering component alone
does not calculate significance or power. The probability-table API below adds
those quantities for single-stage rejection regions.

## Single-stage probability tables

```python
from mdanderson_stats import ksbin2_probability_table

table = ksbin2_probability_table(
    20,
    15,
    probability1=0.6,
    probability2=0.2,
    criteria=(1, 2),
    alternative="greater",
)
print(table.significance, table.power)
print(table.null_grid, table.null_rejection)
```

Each last-axis entry represents an inclusive rejection region ending at a complete
tied group in `ordering.group_end`. Regions grow from the strongest evidence to
the entire outcome space. `power` sums their probabilities under the supplied
alternative probabilities, which broadcast with a final group axis. For a one-sided
test, the supplied probabilities must follow the indicated direction. Endpoints
are supported. Two-sided calculations also allow equal probabilities.

`null_rejection` has shape (grid points, groups), evaluating p1 = p2 at every
null grid point. `significance` is its column maximum. The accompanying
`maximizing_null_probability` identifies the first maximizing grid point; it is
not a fitted continuous maximizer. Defaults reproduce the source's 51-point grid:
0 through 1 by 0.02 for one-sided tests, or 0 through 0.5 by 0.01 for two-sided
tests. The latter exploits complement symmetry of two-sided regions. A custom
strictly increasing grid in [0,1] can be supplied, including a full-range grid.

This grid maximum does not establish the supremum over every possible common null
probability. It should not be reported as a guaranteed continuous-nuisance size
bound. The routine calculates the complete table instead of truncating the source
SSSIG computation above its display significance limit. Ordinary regions include
all outcomes in their terminal tied group; no mid-p adjustment or boundary
randomization is applied.

Binomial probabilities reuse the shared cached-combinatorial/log-weighted kernel.
The two independent group distributions are multiplied in sorted outcome order,
and cumulative sums evaluate every complete rejection region together. The
calculation uses NumPy across both outcomes and probability cases.

`tools/reference_ksbin2_probability.py` extracts unchanged SSSIG, SSPOW, PQTAB,
PQTAB1 and QEQDBL routines. Its independent driver supplies explicit sorted count
pairs, exact-combinatorial scaled coefficients and tied-group endpoints. This
validates the original probability routines separately from the ordering tests;
it is not a native main-program session. Eighteen tables span three sample-size
pairs, all directions and two criterion lists. Source/extracted hashes and compiler
provenance accompany `tests/fixtures/ksbin2_probability.json`. Comparisons pass at
3e-12 relative / 2e-14 absolute tolerance.

Independent tests enumerate all binary paths for two groups of sizes 2 and 3,
including endpoint probabilities, and check every region's null rejection and
power. Other tests check broadcasting, complete tied regions, complement symmetry,
custom grids, the maximum 100-by-100 design, and a power of 1e-300.

## Mid-p convention and rejection-region selection

```python
from mdanderson_stats import ksbin2_probability_table

table = ksbin2_probability_table(20, 15, 0.6, 0.2)
region = table.select(alpha=0.05)
print(region.events, region.significance, region.power)
midp_region = table.select(alpha=0.05, method="midp")
print(midp_region.reported_significance, midp_region.significance)
```

`select` returns the largest complete tied rejection region whose reported level
is at most alpha. Exact equality is included; every group in a qualifying plateau
is included. `select_group` allows explicit zero-based group selection, matching
the source's requirement that a critical boundary end on a whole tied group.
Group -1 denotes the empty region and yields no event pairs, zero significance,
and zero power. Alpha is scalar in [0,1]; alternative-power arrays preserve their
broadcast shape. The returned event pairs describe the complete inclusive region.

The ordinary method uses the existing grid-maximum significance. The `midp` method
reproduces the original BRKARR convention: average the current and preceding
groups' ordinary grid maxima, with zero preceding the first group. It leaves power
and the actual rejection region unchanged, as the source does. The result exposes
both `reported_significance` (the selected convention) and `significance` (ordinary
grid-maximum rejection probability). For one observation per group, selecting the
first greater-tail group at mid-p alpha 0.125 gives reported significance 0.125
but ordinary significance 0.25. Its power is p1*(1-p2), without halving.

For clarity, the table also exposes `null_midp`: at each null probability, the
strict-tail rejection probability plus half the probability of the terminal tied
group. `pointwise_midp_significance` takes the grid maximum of those pointwise
values. This generally differs from `midp_significance`, which adjusts after
maximization. The latter is at least as large because the two adjacent ordinary
maxima may occur at different grid points. These are distinct conventions; the
selection method named `midp` deliberately follows the original executable.
Neither changes the grid's finite coverage into a continuous-null size guarantee.

The native reference tool additionally extracts unchanged BRKARR and records its
mid-p output for all 18 probability-table cases. Independent tests sum strict and
half-weight terminal masses at every grid point in a small design. Selection tests
cover every attainable level (including exact ties), empty/full regions, both
one-sided directions and two-sided tests, explicit groups, broadcast power, and
independent binomial-mass power sums for the chosen inclusive regions.

## Fixed multistage designs

```python
from mdanderson_stats import KStageTwoSampleBinomial

design = KStageTwoSampleBinomial(
    cumulative_trials=[[2, 2], [3, 3], [4, 4]],
    reject_group=[0, 0, 1],
    quit_group=[2, 2],
    criteria=(1,),
    alternative="greater",
)
result = design.operating_characteristics(probability1=[0.5, 0.6], probability2=[0.5, 0.2])
print(result.rejection_probability, result.expected_sample_size)
```

Each stage's rejection and quitting boundaries index its **reachable** tied score
groups, numbered from zero in strongest-to-weakest evidence order. `reject_group`
includes all groups up through that index; `quit_group` includes that group and
all later groups. -1 disables a boundary. Supply a rejection entry per stage and
quitting entries only for interim stages. Final nonrejections always quit.
`orderings[stage-1]` exposes the corresponding reachable count pairs and tied groups.
Changing earlier stopping decisions can change later group indices.

Cumulative sizes have shape (stages, 2), with 1–10 stages and 1–100 observations
per group. Sizes cannot decrease; each stage adds observations to at least one
group. The other group's increment may be zero. Overlapping regions, out-of-range
group indices, and designs eliminating every path to a planned stage raise errors.

The constructor counts surviving paths through separable binomial-coefficient
convolutions. It caches the fraction of paths reaching each cumulative outcome,
then weights the two binomial distributions for each evaluation. This avoids
recomputing transitions for each hypothesis or probability scan. Binary64 path
counts fit within the original total-200-observation domain.

`stage_distribution(stage, p1, p2)` returns joint arrival probabilities with final
axes indexing the two cumulative event counts. Its sum is the probability of
reaching the stage, not a conditional distribution summing to one. Probability
pairs broadcast and may include endpoints, equal probabilities, or effects in the
opposite direction to the design. Stage numbers start at one.

`operating_characteristics` returns rejection, quitting and continuation arrays
with a final stage axis, total rejection probability, and unconditional expected
sample sizes with a final two-group axis. Expected sizes weight cumulative group
sizes by the probability of terminating at each stage. This is evaluation of a
fixed design; it does not search for optimal boundaries or maximize multistage
significance over the nuisance probability.

Validation includes exhaustive enumeration of paired binary sequences in a
three-stage design, comparing every stage's arrival distribution and decisions,
probability conservation and expected sample sizes. Tests cover three directions,
seven probability pairs including endpoints, disabled boundaries, one-group-only
increments, broadcasting, one-stage agreement and invalid designs. Nine native
transition fixtures use the unchanged coefficient update/repacking block from
SSUPD, with independently calculated scaled binomial inputs and helpers. Native
sorting is not invoked in this reference; ordering is validated separately.
Source, kernel and driver hashes accompany `tests/fixtures/ksbin2_transition.json`.

## Multistage boundary assistance and power loss

```python
from mdanderson_stats import (
    KStageTwoSampleBinomial,
    ksbin2_boundary_table,
    ksbin2_probability_table,
)

design = KStageTwoSampleBinomial(
    [[2, 2], [3, 3], [4, 4]],
    [0, 0, 1],
    [2, 2],
    criteria=(1,),
)
reference = ksbin2_probability_table(4, 4, 0.6, 0.2, criteria=(1,)).select_group(2)
table = ksbin2_boundary_table(design, 2, 0.6, 0.2, reference=reference)
print(table.significance, table.cumulative_power, table.power_loss)
```

The table evaluates every complete tied rejection region reachable at the requested
stage. `stage_power` and `stage_null_rejection` are the current stage's contributions;
`previous_power` and `previous_null_rejection` sum earlier rejections. Cumulative
values add these before taking the null-grid maximum, so `significance` is the
maximum cumulative rejection probability across the grid, not a sum of separate
stage maxima. Future stopping decisions do not affect these quantities. As with
the fixed-design evaluator, supplied probabilities may represent nulls, endpoints,
or effects opposite to the design direction.

Default and custom null grids follow the single-stage table conventions. All
significance values here use ordinary inclusive regions. Multistage mid-p display
semantics remain pending; this table does not silently apply the single-stage
mid-p formula to a nonzero prior rejection probability.

An optional `reference` supplies a fixed single-stage rejection region with the
same final sample sizes. Its explicit event pairs define rejection; its stored
probabilities are not reused. Reference completion is evaluated at the probability
pairs supplied to the boundary-table call. The reference can have a different
ordering or direction, since its complete fixed event region is unambiguous.

`conditional_reference_power` has one last-axis entry per reachable **outcome row**,
representing its probability of eventually landing in that reference region if
all remaining observations are collected. This uses independent binomial transition
matrices and sums over the reference event mask. Counts outside the feasible
increment range contribute zero; zero remaining observations give an identity
transition. Earlier stopping affects arrival probabilities, while future stopping
is deliberately ignored in reference completion.

`power_loss` has one last-axis entry per tied **group**. It sums alternative arrival
mass times conditional reference power over that group's inclusive futility region
(the group and all weaker-evidence groups). This is absolute probability, not a
fraction of reference power or a randomized decision. With no reference, both
reference-completion fields are None. These are design-assistance quantities, not
an exact power difference between arbitrary multistage designs.

Validation independently enumerates all paired four-observation sequences in 27
cases spanning stages, directions and probability endpoints. Every candidate
region's stage/cumulative probabilities and reference power loss are checked.
Additional checks cover the native-validated first-stage table, empty/full reference
regions, broadcasting, a known conditional completion probability, no new data in
one group, and invalid arguments. The implementation follows the SSSIG/SSPOW/SSPL
probability definitions; these tests are exhaustive probability checks rather than
a native main-program session or a separate native SSPL reference build.
