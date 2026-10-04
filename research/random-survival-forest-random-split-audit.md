# RF-SRC random survival split audit

This audit pins the `split_rule="random"` split kernel exposed by the Python
survival forest. Cached native sources are randomForestSRC 3.2.2, identified in
[`random-survival-forest-audit.md`](random-survival-forest-audit.md) by the
CRAN Git mirror tag and source blob hashes. No native random stream or complete
`get.brier.survival` pipeline parity is claimed.

## Split rule contract

`src/randomForestSRC.c:12247-12390` implements `randomSplitGeneric`.
`getPreSplitResultGeneric` at `:21795-21822` requires a parent membership
count of at least `2*RF_nodeSize`; the split loop in `randomSplitGeneric`
accepts any candidate with nonzero left and right sizes (`:12347-12365`).
Thus `nodesize` constrains eligible parents, not the minimum size of each
daughter.

The generic feature selector at `:22067-22090` increments its `mtry`
counter for every selected variable and removes that variable from the
selection distribution. A selected feature with fewer than two distinct
nonmissing values is marked impermissible in phase 1 (`:22092` onward),
consumes its slot, and allows the next selected feature to be tried. Once a
valid candidate updates the split record, the outer loop ends because its
condition requires the current score to remain NaN (`:12297-12303`).

For the unweighted, no-missing native fast path, selection is specialized by
`selectRandomCovariatesSimpleSingle` (`:22806-22843`): with `mtry==1` it draws a
permissible variable even when there is only one; with `mtry>1`, if `mtry`
covers the permissible set it takes features in stored order without random
draws, otherwise it draws them sequentially without replacement. The
`rfsrc.R` wrapper defaults survival `mtry` to `ceil(sqrt(p))`, documented in
`man/rfsrc.Rd:119-122`. The Python API has finite covariates and no feature
weights, so this simple/no-missing feature-selection branch is the applicable
contract; the weighted/missing generic-distribution variants are outside this
API's claim.

The split objective is always zero for a nonempty two-sided candidate and NaN
otherwise (`randomSplitGeneric`, `:12347-12353`). Consequently, candidate
optimization and score ties do not select the split. The first valid
candidate in sequential feature-selection order wins. Under `RAND_SPLIT`,
phase 2 sets exactly one candidate for continuous and factor features
(`:22297-22305`, `:22364-22392`), regardless of the R caller's `nsplit=1`.
For a continuous feature with node-unique sorted values
`x_(1) < ... < x_(k)`, the cut is sampled uniformly from `x_(1),...,x_(k-1)`
(`:22388-22392`). Routing uses `x <= cut` (the prediction kernel at
`:7819`); cuts are observed values, not midpoints.

For factors, `getRandomPair` (`:22639-22669`) chooses a cardinality group with
probability proportional to that group's number of distinct unordered
complementary partitions. `createRandomBinaryPair` (`:22671-22699`) samples
the indicated number of observed levels uniformly without replacement.
`makeFactor` computes group counts as choose(K,k), halving the balanced
K/2 group to count complements only once (`:3350-3377`). Therefore every
unordered nontrivial partition has equal probability. The Python helper uses
log-binomial weights to avoid integer/float overflow and then performs the
same group-then-subset draw. Level codes are the sorted observed levels at the
node; packed routing stores the chosen left subset.

## Censoring-model call settings

Cached `R/utilities.survival.R:356-371` constructs a secondary survival
outcome from the original time and censoring indicator, then calls
`rfsrc(..., ntree=50, nsplit=1, splitrule="random", nodesize=set.nodesize(n,p),
perf.type="none")` and predicts its survival curves on the requested profiles.
The helper at `:232-247` sets `nodesize=2` for `n<=300,p>n`, `5` for
`n<=300,p<=n`, `10` for `300<n<=2000`, and `n/200` otherwise. `rfsrc.R:491`
casts this supplied value to integer, so the last branch truncates when
nonintegral. This differs from the generic RF-SRC survival default
`nodesize=15` (`man/rfsrc.Rd:128-131`) and must be passed explicitly for that
route. The wrapper's default no-replacement sample size is `round(.632*n)`
(`rfsrc.R:9,195-212`; R uses ties-to-even rounding); Python's default uses
`np.rint` for the same sample count. `mtry` is not supplied at this call, so
the survival default `ceil(sqrt(p))` applies.

## Python boundary and validation plan

The implementation branches only for explicit `split_rule="random"`; the
existing `logrank`, `logrankscore`, and `bs.gradient` candidate scoring and
draw paths are left intact. Python's NumPy generator is seeded and replayable
but does not reproduce RF-SRC's R/C stream. The fit returns its usual forest
object with `split_rule="random"` and no Brier `split_probability`.

The focused tests ledger the random cut independently of survival outcomes,
constant-feature `mtry` consumption, the full-`mtry` ordered fast path,
categorical group probabilities and group/subset draw order, and the public
fact that `nsplit` does not alter the random split candidate count. Existing
default-mode fixtures remain the regression guard. No native forest, native
random-number stream, or censoring-distribution evaluator is introduced in
this change.
