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
When no missing rows were observed, an exported missing branch can inherit
its parent's stored weight and prediction. That weight is a fallback value,
not an observed missing-row count.

The native wrapper rejects a training sample when
`n_train * bag_fraction <= 2 * n.minobsinnode + 1`. With the CondiS defaults,
each training fold therefore needs at least 43 rows. This check occurs before
native fitting. It must not be silently bypassed by changing bag fractions or
node sizes. Small fixtures suitable for ridge or neural fits may be unsuitable
for this learner.

## Executed native references

`tools/reference_condis_gbm.R` compiles the complete unchanged native source
directory with one build job, sources the original R fit/predict helpers and
calls the original caret fit wrapper with its namespace dispatch redirected
to the source-loaded `gbm.fit`. There is no package installation. It manually
applies the verified grid and shared-fold aggregation, not the entire caret
training pipeline.

`tests/fixtures/condis-gbm-inputs.json` retains two deterministic synthetic
cases: 80 rows with three covariates and 72 rows with 60 covariates. Status is
an additional predictor. Their imputed targets come from the previously
verified Python base CondiS; the native comparisons exercise refinement
conditional on those fixed targets. Both use four explicit folds large
enough to satisfy the native default sample-size rule.

`tests/fixtures/condis-gbm-native.json` saves all nine fold-score surfaces,
150-tree full-data paths for each depth, and a separate selected-model refit.
Smaller candidate tree counts use prefixes of the same path. The ordinary
case selects 100 trees and depth 2, with mean fold RMSE 4.3196391411639343.
The wide case selects 50 trees and depth 3, with mean fold RMSE
4.2965521343784872. These values describe the recorded native seeds; they do
not claim equal-seed R/NumPy RNG parity.

An additional three-tree, depth-three fit for each case retains every tree
component and its uniform subsampling draws. The script verifies that the
actual native fit leaves exactly the same RNG state as drawing one uniform
per row per tree. It also executes the native sample-size guard: 42 rows
are rejected with defaults, while 43 rows produce finite predictions.

Serial compilation and reference generation passed in 12.11 seconds, with
105.6 MiB peak child resident memory and zero swaps. Independent NumPy
reconstruction of the subsamples, tree traversal and cumulative predictions
matched all saved three-tree prefixes exactly. Recomputed nonmissing node
predictions differed by at most 2.23e-16 absolute, trace training errors by
3.56e-15, and full-path training errors by 5.33e-15. Fold RMSE reconstruction
matched exactly. The checks also verified the inherited fallback metadata of
11 empty missing branches rather than incorrectly interpreting those weights
as sample counts.

## Python work still required

These results establish the native reference data and algorithm contract;
they do not implement or validate Python tree construction. Equal R and
NumPy seeds are not interchangeable. The retained draw streams support direct
construction comparisons independently of the random-number generator.

The Python implementation should reuse sorted predictor orders, grow trees
sequentially and accumulate predictions at the requested tree counts. It must
not allocate an observations-by-trees tensor for ordinary refinement. Stable
residual sums, split gains and midpoints need numerical checks against the
original objective. Tree-prefix reuse, sample-size rejection and exact event
restoration are meaningful validation targets; native reference generation
does not belong in ordinary CI.
