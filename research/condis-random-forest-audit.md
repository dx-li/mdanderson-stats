# CondiS-X regression-forest source contract

This is preparation for the remaining random-forest learner, not implemented
coverage. The Python package does not yet expose this CondiS-X learner.
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

## Python work still required

The references validate native calculations and prepare direct numerical
checks. They do not constitute a Python forest learner, random-stream replay
or a completed CondiS-X refinement. A future implementation must verify
construction and RNG consumption against the traces, including duplicate
features and split ties, before claiming tree parity.

The Python implementation should grow trees sequentially, reuse per-tree
work arrays and accumulate predictions without an observations-by-trees
matrix unless explicitly requested. Stable centered split calculations and
midpoints should avoid overflow while preserving the native criterion. Those
are implementation requirements, not completed performance or parity claims.
