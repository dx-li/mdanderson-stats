# CondiS-X regression-forest source contract

This is preparation for the remaining random-forest learner, not implemented
coverage. The Python package does not yet expose this CondiS-X learner.
[`condis-random-forest-sources.json`](condis-random-forest-sources.json) pins
nine files from randomForest 4.7-1.2 at commit
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

## Validation work still required

The retrieved regression compilation closure is `regrf.c`, `regTree.c`,
`rfutils.c` and `rf.h`, together with R's runtime. It has not yet been compiled
or executed for this learner. A source-only reference harness should exercise
the original wrapper and native kernels, with shared bootstrap/candidate/tie
draws when checking exact trees. R and NumPy seeds alone do not provide the
same random stream. Compact cases should distinguish all-tree predictions
from out-of-bag predictions and exercise rounded `mtry`, small terminal nodes,
duplicate features and split ties.

The Python implementation should grow trees sequentially, reuse per-tree
work arrays and accumulate predictions without an observations-by-trees
matrix unless explicitly requested. Stable centered split calculations and
midpoints should avoid overflow while preserving the native criterion. Those
are implementation requirements, not completed performance or parity claims.
