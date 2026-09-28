# CondiS-X regression-forest source contract

The Python package implements the numeric random-forest CondiS-X learner.
[`condis-random-forest-sources.json`](condis-random-forest-sources.json) pins
ten files from randomForest 4.7-1.2 at commit
`0ad64d71e886bff5b241cbc31a3240bf0e822914`. The downloaded files were verified
against their Git blob hashes and have recorded SHA-256 digests. This is a
current source reference, not proof of the dependency version used in 2022.

## CondiS call and forest defaults

The original `CondiS-X.R` uses `pred_time ~ .`, so the design includes event
status as well as the supplied covariates. It supplies one tuning setting:
`mtry = sqrt(ncol(covariates))`. The denominator here excludes the status
column. The randomForest wrapper rounds that value and clamps it between one
and the number of actual predictors; truncating the square root is incorrect.
The existing CondiS cross-validation contract still applies even though this
grid has only one setting.

The native regression defaults are 500 trees, uniform bootstrap samples of
the full training-row count with replacement, and `nodesize = 5`. Predictors
are not scaled. The R wrapper centers the response before native fitting and
adds its mean back to every stored terminal-node prediction. Bias correction,
permutation importance and proximity calculation are off by default.

CondiS explicitly predicts with its data supplied as `newdata`. This returns
the mean over **all trees**, including trees trained on each row. Calling the
native prediction method with no `newdata` instead returns out-of-bag fitted
values; those are not the CondiS refinement. Observed event times are then
restored by the surrounding CondiS code.

## Tree construction details

`src/regrf.c` performs bootstrap sampling, grows each tree, and aggregates its
predictions. For the unweighted default, each sampled index is
`floor(unif_rand() * n)`. `src/regTree.c` contains the regression split and
prediction algorithms:

- Nodes are processed in allocation order; each successful split appends a
  left and right child. Predictor candidates are sampled without replacement
  independently at each node using swaps in an index vector.
- Numeric split candidates are gaps between unequal sorted feature values.
  The criterion is `sum_left**2 / n_left + sum_right**2 / n_right -
  sum_parent**2 / n_parent`, equivalent to the decrease in squared error.
  The split point is the midpoint, and values equal to it go left.
- A child containing at most `nodesize` bootstrap rows is terminal. This is
  a rule for whether a child is split again, **not a minimum child size for
  the current split**. The root is initially marked for splitting, even for
  a small training sample.
- Equal split criteria use random tie decisions. The source uses consecutive
  `if` statements for a new maximum and equality, so even a newly found
  maximum consumes a tie draw. A port aiming to replay a supplied random
  stream must retain this order. It must also check the native zero-gain
  behavior rather than assume every chosen split has strictly positive gain.
- Terminal predictions are node response means. The forest mean uses equal
  tree weights. Bootstrap multiplicities participate in node counts and
  means; replacing the bootstrap sample by unique rows changes the method.

Native categorical splits order levels by their response means and store a
packed level mask. The existing Python CondiS interface accepts numeric
covariates; a later categorical extension must make its encoding explicit.

## Native reference calculations

The retrieved regression compilation closure is `regrf.c`, `regTree.c`,
`rfutils.c` and `rf.h`, together with R's runtime. The executed
`tools/reference_condis_rf.R` compiles these unchanged files locally, sources
the original R fit/predict wrappers and calls the original caret fit wrapper
with the CondiS setting. Only namespace dispatch is redirected to the
source-loaded function. There is no package installation and no complete
caret training-pipeline execution. The shared folds and manual fold-RMSE
aggregation follow the previously verified CondiS contract.

`tests/fixtures/condis-rf-native.json` records the ordinary and wide cases
using 500 trees per fold and full-data fit. Requested `mtry` values are
`sqrt(3)` and `sqrt(20)`, which become 2 and 4, respectively. Mean fold RMSEs
are 2.4777680157042048 and 3.4566954745997167. Native seeds are recorded for
reproducibility; equal R and NumPy seeds alone do not supply the same stream.

To support direct construction checks, an additional five-tree forest per
case retains its structure, split thresholds, node means, bootstrap counts,
terminal-node assignments and individual predictions. A small C recorder
intercepts the kernels' uniform-RNG calls at compilation and returns each
original R draw unchanged. These traces contain 752 and 434 draws, including
bootstrap, predictor sampling and tie decisions. The script checks that
turning recording off with the same seed preserves bootstrap counts and
predictions exactly. The recorder's R-owned buffer is retained throughout
the fit and cleared before release.

Compilation and reference generation passed in 1.96 seconds using one build
job and one R process, with 146.9 MiB peak child resident memory and zero
swaps. Independent NumPy traversal of all ten retained trees reproduced
terminal-node assignments and individual/forest predictions exactly.
Recomputed node means, weighted by bootstrap multiplicities, differed by at
most 3.56e-15 absolute. Fold RMSE reconstruction differed by at most 8.89e-16.
The first tree's bootstrap counts also matched the recorded uniform draws.

Terminal samples in these forests range from one to six rows. This confirms
that `nodesize = 5` is not a minimum child size; a larger node may also remain
terminal when the sampled predictors do not provide a split. All-tree and
out-of-bag predictions differ by as much as 3.9113 and 4.0200 on these inputs,
so the fixture distinguishes those outputs clearly.

## Python implementation and numerical limits

`condis_forest.py` implements numeric regression trees with bootstrap
multiplicities, sampled predictors, native-style split/tie draws, terminal
node means and equal-weight all-tree predictions. The CondiS wrapper adds
status once, derives `mtry` from the original covariate count, computes fold
RMSE for the single native setting and fits a fresh full-data forest.
Standalone fit/predict functions accept an explicit predictor matrix.
Trees are stored compactly; optional in-bag and per-tree prediction matrices
are bounded. Default prediction uses one working vector.

Using the original recorded uniforms, the first three ordinary trees have
exactly matching node counts (25, 29, 29), split variables, thresholds and
bootstrap multiplicities. Individual predictions differ by at most 8.9e-16.
The next tree first differs at zero-based node 22, where two observations
with responses 12.1 and 6.76666667 can be separated by several predictors
with the same positive gain. A different tie path changes the selected
feature and later random consumption. This is not a complete native RNG or
tree-identity claim. Python also stops a node if a degenerate zero-gain
candidate would create an empty child, rather than forcing the native
ordering-dependent partition.

Centered responses are rescaled when needed to protect split gains at tiny
and large magnitudes. The `<=` left-branch threshold remains below the upper
observation when adjacent floats have no interior midpoint. Fold RMSE and
its mean/sample standard deviation use normalized arithmetic to avoid
squared-error underflow and variance overflow.

Three focused checks passed: direct native three-tree construction,
tiny-response/adjacent-float and constant-response behavior, and the wrapper's
status/mtry rules and RMSE scale invariance at 1e-200 and 1e200. Targeted Ruff
and mypy passed. A bounded ordinary 48-row, ten-fold workflow using 500 trees
per fit took 2.214 seconds and produced finite outputs, with roughly 610,000
split-gap evaluations. Mean RMSE was about 2.48573 versus the saved native
2.47777; the runs use different random streams and this is not a numerical
parity tolerance. No large simulation or additional CI workflow was added.

After public API integration, the three focused forest tests passed in
1.20 seconds. Targeted lint, formatting and type checks passed, and the
documented example reproduced full-fit predictions exactly through the
public prediction function.
