# CondiS-X Gaussian gradient-boosting source contract

The gradient-boosting CondiS-X refinement remains unimplemented in Python.
[`condis-gbm-sources.json`](condis-gbm-sources.json) records 60 hash-verified
original gbm 2.3.1 files at commit
`638b95f74de63e5d9c0212aeb11c38c08608c14b`, including the complete native
source directory and the R fitting/prediction helpers. This version is a
source reference; it is not asserted to be the dependency installed when
CondiS was published in 2022. The original license is GPL version 2 or later.

## CondiS and caret settings

`CondiS-X.R` passes `pred_time ~ .` to caret's `gbm` model, using event status
alongside the supplied covariates. Its computed preprocessing object is not
passed to training. The pinned caret model metadata was loaded and inspected
for its grid, loop, fit, prediction and sort functions; those establish:

- Numeric imputed times select the Gaussian distribution. The fit wrapper
  calls `gbm::gbm.fit` with predictor data, response and the tuning settings.
- Default tuning length three gives tree counts 50, 100 and 150 crossed with
  interaction depths 1, 2 and 3. Shrinkage is fixed at 0.1 and minimum node
  observations at 10, for nine settings in total.
- Selection order is increasing tree count, then interaction depth, then
  shrinkage. The default bag fraction inherited from `gbm.fit` is 0.5.
- Caret groups settings by depth, shrinkage and minimum node observations.
  It fits a 150-tree sequence for each depth and obtains the 50- and 100-tree
  predictions as prefixes of that fit. Training an independent randomized
  model for every candidate would change the native cross-validation path.
- The surrounding CondiS fold-RMSE selection, full-data refit and restoration
  of observed event times apply here too. These are imputed-response scores,
  not an independent validation of the entire survival-imputation pipeline.

## Native Gaussian boosting algorithm

`src/gaussian.cpp` initializes the model at the training-response mean. Each
tree fits residuals `y - fitted`; the Gaussian terminal-node mean is already
the best squared-error constant and needs no additional optimization. The
update adds shrinkage times the tree prediction to **all training rows**.
The reported Gaussian deviance is weighted mean squared error.

`src/gbm_engine.cpp` takes exactly `floor(bag_fraction * n_train)` rows
without replacement on each iteration. It scans every training row in order
and accepts it when `uniform * remaining_rows < remaining_slots`, consuming
one uniform draw per row even after the bag becomes full. This is a fresh
subsample for each tree, not bootstrap sampling with replacement.

`src/tree.cpp` grows a tree by selecting the available terminal node with
the greatest split improvement. Its `interaction.depth` parameter limits
the number of split steps in this best-first loop, rather than defining a
complete binary tree with that many levels. Each split allocates left, right
and missing branches. All predictors are scanned in original column order.
Strict improvement comparisons retain the first equal candidate; there is
no random predictor subsampling or random split-tie resolution on this path.

For numeric covariates, `src/node_search.cpp` considers midpoints between
distinct sorted values and requires at least `n.minobsinnode` sampled rows
on each nonmissing side. Prediction uses **strictly less than the split
point for the left branch**, and equal values go right. This differs from
the regression-forest branch. `src/node_nonterminal.cpp` fills a missing
branch with too few samples from the weighted mean of its nonmissing
children. Stored R tree predictions already include the shrinkage factor;
an independent predictor must avoid applying it twice.

The native wrapper rejects a training sample when
`n_train * bag_fraction <= 2 * n.minobsinnode + 1`. With the CondiS defaults,
each training fold therefore needs at least 43 rows. This check occurs before
native fitting. It must not be silently bypassed by changing bag fractions or
node sizes. Small fixtures suitable for ridge or neural fits may be unsuitable
for this learner.

## Next implementation and validation steps

The complete C/C++ compilation sources have been retrieved, but this
checkpoint has not compiled or executed the boosting kernel. The next native
reference should use sufficiently large explicit shared folds, preserve
caret's shared 150-tree prefix path, and retain small tree sequences with
their subsampling draws for direct construction checks. Record the original
seed and distinguish that from a shared draw stream; equal R and NumPy seeds
are not interchangeable.

The Python implementation should reuse sorted predictor orders, grow trees
sequentially and accumulate predictions at the requested tree counts. It must
not allocate an observations-by-trees tensor for ordinary refinement. Stable
residual sums, split gains and midpoints need numerical checks against the
original objective. Tree-prefix reuse, sample-size rejection and exact event
restoration are meaningful validation targets; native reference generation
does not belong in ordinary CI.
