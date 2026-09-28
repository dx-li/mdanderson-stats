# CondiS censored-lifetime imputation

`condis_impute` implements the base CondiS method in MD Anderson catalog entry 157,
[CondiS](https://biostatistics.mdanderson.org/shinyapps/CondiS/).
The method is described by Wang, Flowers, Li and Huang (2022),
*Journal of Biomedical Informatics* 131:104117,
[doi:10.1016/j.jbi.2022.104117](https://doi.org/10.1016/j.jbi.2022.104117).
The [CRAN package](https://cran.r-project.org/package=CondiS) version 0.1.2 was
reviewed and executed for numerical comparisons.
[Source provenance](condis-sources.json) records the archive. Original R programs
and data are not redistributed.

## Calculation

Let c be an observed right-censoring time, S the estimated survival curve, and h
the chosen horizon. For c<h, the imputation is

`c + integral(c, h, S(t) dt) / S(c)`.

Observed event times are preserved exactly. A censoring time at or beyond h is
also preserved. The default horizon is maximum observed follow-up. Explicit
horizons must be nonnegative and no greater than that maximum; no unsupported
tail extrapolation is performed. This is a restricted conditional mean, not an
estimate of an unrestricted lifetime beyond available follow-up. Preserving times
past h follows the native function; this does not truncate every subject to h.

The underlying Kaplan–Meier calculation reuses `exploratory_survival` with exact
ties grouped and tied censorings included in the risk set for tied events.
Inputs are complete vectors with nonnegative finite times and status
`1=event, 0=right-censored`. Input order is retained in the output.

Two interpolation conventions are available:

- `interpolation="linear"` (default): straight lines connect right-continuous
  Kaplan–Meier values at distinct observed times, matching the R function's
  `approxfun` convention. Areas are evaluated analytically by trapezoids.
- `interpolation="step"`: integrate the actual right-continuous Kaplan–Meier
  step function using rectangles. This is an explicit alternative to native
  interpolation; the two methods need not yield the same imputed times.

One sorted curve and reverse cumulative interval integrals handle every censored
observation. Time widths are normalized by the horizon before summation, with
normalization after subtraction to preserve close time differences. Complexity is
O(n log n) for sorting and O(n) storage; no per-subject adaptive integration or
n-by-n matrix is needed. Up to one million observations are supported.

## Usage

```python
import numpy as np
from mdanderson_stats import condis_impute

fit = condis_impute([1, 2, 4, 6], [1, 0, 1, 0])
np.testing.assert_allclose(fit.imputed_time, [1, 4.5, 4, 6])
step = condis_impute([1, 2, 4, 6], [1, 0, 1, 0], interpolation="step")
np.testing.assert_allclose(step.imputed_time, [1, 5, 4, 6])
restricted = condis_impute([1, 2, 4, 6], [1, 0, 1, 0], horizon=3)
np.testing.assert_allclose(restricted.imputed_time, [1, 2.875, 4, 6])
```

`CondiSImputation` retains observed times, binary status, imputed times, added
remaining times, horizon, interpolation mode and the fitted survival curve.
Array results are read-only. Added remaining time is zero for observed events and
for censoring at/beyond the horizon. Degenerate one-time samples and all-censored
samples have well-defined restricted results: all-censored subjects before the
horizon receive that horizon. These degenerate cases are not a native runtime
parity claim.

The estimated curve assumes independent censoring for the population being
analyzed. Imputed targets are deterministic estimates, not observed failures and
not multiple-imputation draws. For predictive evaluation, estimate imputations
within the training data/folds rather than using held-out survival outcomes to
construct training targets. This function does not provide a fitted transform
for applying a training curve to a separate held-out sample.

## Default CondiS-X linear refinement

`condis_linear_refine(imputation, covariates)` adds the native default `glm`
refinement. Its regression uses an intercept, the **censoring status column**,
and every supplied numeric covariate column to predict the base imputed times.
The native source includes status through its `pred_time ~ .` formula. After
fitting on all supplied rows, observed event times are restored unchanged.
Supply covariates in the same row order, with categorical factors explicitly
encoded as numeric columns.

The verified caret model uses a Gaussian family with identity link. There is no
hyperparameter to tune for this model: a scaled least-squares solve reproduces
the final full-sample fit without running repeated cross-validation. Native
resampling metrics are not returned. Column centering/scaling with an intercept
preserves fitted values; a rank-revealing solve handles dependent or constant
columns. Responses are scaled to support very small or large time units. The
implementation accepts up to 500 covariate columns and 2 million covariate cells.

```python
from mdanderson_stats import condis_linear_refine

base = condis_impute(
    [8, 1, 2, 2, 4, 6, 10, 10, 12, 15],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
)
refined = condis_linear_refine(base, np.zeros((10, 1)))
assert refined.below_censoring[-1]
bounded = condis_linear_refine(base, np.zeros((10, 1)), enforce_censoring=True)
assert bounded.refined_time[-1] == 15
```

`CondiSLinearRefinement` retains raw `fitted_time` for all rows, `refined_time`
after event restoration and optional clipping, raw censored-prediction flags
`below_censoring` and `above_horizon`, design rank, residual degrees of freedom,
and training residual RMSE against base imputed targets. This RMSE is not an
out-of-sample prediction error or uncertainty estimate.

As in native CondiS-X, unconstrained fitted times can be negative, below a known
censoring time, or above the base horizon. The default returns those predictions
with diagnostic flags. `enforce_censoring=True` is an explicit Python extension
that clips censored predictions at their observed lower bound. It does not cap
them at the horizon, refit the model, or change the raw predictions/flags. This is
a refinement of the supplied sample, not a deployable survival prediction model:
future censoring status is generally unavailable for new patients.

## Ridge, lasso and nearest-neighbor refinement

`condis_regularized_refine(imputation, covariates, method=...)` tunes and refits
three additional CondiS-X learners. As in the linear refinement, the predictors
include status and every supplied numeric covariate, and observed events are
restored after prediction. These are in-sample refinements of imputed times.

| Method | Default tuning grid | Predictor handling |
| --- | --- | --- |
| `"ridge"` | 10 lambdas spaced linearly from 0.01 to 10 | Gaussian glmnet-style standardized path, alpha=0 |
| `"lasso"` | Same lambda grid | Gaussian glmnet-style standardized path, alpha=1 |
| `"knn"` | k=5,7,9 | Raw Euclidean distances, including status; mean neighbor response |

The default is ten folds and one repeat, matching the original
`trainControl(method="repeatedcv")` defaults. `folds`, `repeats` and
`random_state` make the resampling explicit. Native R and NumPy random streams
are different; equal seeds do not imply equal fold assignments. Supply
`fold_ids` to reuse or externally specify held-out folds, with zero-based labels
and shape `(repeats, n)` or `(n,)` for one repeat.
For fewer observations than requested folds, the effective fold count is the
sample size. At least two observations are required.

Tuning minimizes the unweighted mean of fold RMSEs. Equal scores select the
larger lambda or larger k. The chosen learner is then fitted to all supplied
rows. These RMSEs measure errors against the supplied imputed targets; because
the base imputation precedes learner tuning, they do not estimate held-out
survival prediction performance for the complete pipeline.

```python
from mdanderson_stats import condis_regularized_refine

row = np.arange(30)
time = 2 + (row * 7 % 31) / 2
status = (row % 4 != 0).astype(int)
covariates = np.column_stack((np.sin(row / 3), row % 5))
base = condis_impute(time, status)
refinements = {
    method: condis_regularized_refine(
        base, covariates, method=method, folds=5, random_state=7,
    )
    for method in ("ridge", "lasso", "knn")
}
selected_settings = {name: result.best_value for name, result in refinements.items()}
```

`CondiSRegularizedRefinement` retains raw fitted times, refined times,
censoring-bound diagnostics, sorted tuning values, mean and sample-SD fold
RMSEs, individual `fold_rmse` values with shape `(repeats, effective_folds,
n_candidates)`, the selected index/value, fold assignments and seed metadata. As for the
linear method, `enforce_censoring=True` explicitly clips censored predictions
at their observed lower bounds after model selection, without changing raw
fits, tuning scores or diagnostic flags. No upper-horizon clipping is applied.

For ridge/lasso, `lambda_grid` replaces the requested tuning values. The
underlying path is automatic: up to 100 points, native response and predictor
scaling, native early-stopping rules, and linear coefficient interpolation at
each requested lambda. Values outside that fitted path are clamped to its
endpoints. This is different from solving only at the requested lambdas. The
default penalty grid depends on the units of time; to compare a change of units,
multiply the explicit `lambda_grid` by the same time conversion factor.

For kNN, `neighbor_grid` replaces the k values. Raw predictor units affect
distances: CondiS computes a preprocessing object but does not use it in the
learner fit. Full-sample predictions include each observation itself. The
implementation follows caret's retained-neighbor tie convention, whose
near-tie behavior can depend on input order. Numerical provenance and details
are recorded in the [learner audit](../research/condis-refinement-audit.md).
Values of k larger than a training sample are clipped to its size, with a
warning when this occurs in cross-validation.

The implementations bound computation rather than launching parallel workers.
Inputs allow up to 500 covariates and 2 million covariate cells, at most 100
requested folds, ten repeats and 100 tuning values. Additional bounds account
for distance and penalty-path work; oversized requests raise an error.
Regularized fits also have bounded coordinate iterations and raise on failure
to converge. kNN requests require k below 1,000 and reject tie sets exceeding
the native retained-neighbor buffer. Constant responses or designs with no
varying predictors produce finite intercept-only mean predictions for ridge
and lasso; this well-defined extension is not a native degenerate-fit parity claim.

## Neural refinement

`condis_neural_refine(imputation, covariates)` fits the CondiS-X neural learner:
one logistic hidden layer and a linear output. Cross-validation compares all
nine combinations of hidden sizes 1, 3 and 5 with weight decays 0.1, 0.0001
and 0, then refits the selected combination on the complete sample. Predictors
include status and the original covariates. The objective is the **sum** of
squared errors plus decay times the sum of squared weights, including biases.
Covariate and response units therefore affect the optimization and penalty.

```python
from mdanderson_stats import condis_neural_refine

neural = condis_neural_refine(base, covariates, folds=3, random_state=7)
np.testing.assert_array_equal(neural.refined_time[status == 1], time[status == 1])
selected_network = neural.selected_hidden_size, neural.selected_decay
iteration_limit_reached = neural.selected_fit.convergence_code == 1
```

The result retains the grid, fold assignments, fold/mean/SD RMSEs, selected
network, fitted weights and predictions, fold objective values, iteration
counts and convergence codes. Observed events are restored after prediction.
`enforce_censoring=True` clips censored predictions at their observed lower
bounds; raw predictions and bound flags remain available. As with the other
learners, CV scores use already-imputed targets and do not measure held-out
performance of the complete survival-imputation pipeline.

The optimizer follows R's inverse-Hessian BFGS procedure. Its default
`max_iterations=100` often reaches the iteration limit in both the native
reference and Python: code 1 records that outcome. Code 0 is the native
stopping status, not a certificate of a global optimum. With zero iterations,
the native code is also 0, but `selected_fit.status` is `"not_run"`.
Other readable statuses are `"stopped"` and `"iteration_limit"`.

This is a nonconvex fit. Native objectives and gradients at fixed weights,
and early optimizer steps, agree closely with Python; small arithmetic
differences can later lead to substantially different fitted networks and
tuning scores, even with identical starts. The [numerical audit](../research/condis-refinement-audit.md)
records a case that agrees through roughly 40 iterations and then diverges.
Neither exact final-fit parity with R nor a globally optimal fit is claimed.

`random_state` gives reproducible Python starts and folds, without reproducing
R's random stream. `fold_ids`, `initial_weights` and `fold_initial_weights`
allow explicit control. Full starts are a nine-element sequence of vectors;
fold starts are nested fold-by-candidate sequences. Candidate order is size
ascending, then decay descending. Each vector has `(p + 1) * size + size + 1`
weights, where `p` includes the status predictor. `fit_condis_neural` exposes
a single fit with explicit size, decay, starts and stopping tolerances.

Fits run serially, with at most 2,000 rows and the native 1,000-weight limit.
Cross-validation also enforces a bounded work budget. These bounds avoid
unbounded allocations; oversized requests raise a clear error.

## Radial support-vector refinement

`condis_svm_refine(imputation, covariates)` tunes and refits CondiS-X's
radial-kernel epsilon-SVR learner. Its default cost grid is 0.25, 0.5 and 1,
with epsilon=0.1. It estimates a single kernel bandwidth before resampling,
then uses the same bandwidth across folds and the final fit. Equal mean fold
RMSEs select the smaller cost. The result restores observed events and exposes
the same optional censoring-bound clipping as the other refinements.

```python
from mdanderson_stats import condis_svm_refine

svm = condis_svm_refine(base, covariates, folds=5, random_state=7)
np.testing.assert_array_equal(
    svm.refined_time[status == 1], time[status == 1],
)
selected_cost, estimated_sigma = svm.best_cost, svm.sigma
```

The automatic bandwidth follows kernlab's sampled-distance rule: draw two
vectors of `floor(n/2)` row indices with replacement, discard zero pair
distances, and average the reciprocals of the 0.1 and 0.9 squared-distance
quantiles. Supply `sigma_pair_indices` with shape `(n//2, 2)` to reproduce
particular pairs, or a positive `sigma` to bypass estimation. These controls
are mutually exclusive. If every sampled distance is zero, provide an
explicit bandwidth or different pairs. Seeds are reproducible within Python;
they do not reproduce R's random stream.

Training normally standardizes predictors and response using sample standard
deviations. **If any predictor is constant, native kernlab disables all
predictor and response scaling for that fit.** This includes the status
predictor. Bandwidth estimation independently follows the same predictor
rule. Consequently, scaling can differ between folds. In the unscaled branch,
raw covariate units affect distances and raw time units affect the meaning of
cost and epsilon. A global change of time units is not an invariance claim
when some folds scale their response and others do not.

`cost_grid`, `epsilon`, `fold_ids`, `folds`, `repeats` and `random_state` expose
the tuning choices. The result retains per-fold scores and solver diagnostics,
the chosen cost, full-sample predictions and censoring-bound flags. These
scores compare predictions with already-imputed targets; they do not measure
held-out performance of the complete survival-imputation pipeline.

The Python solver uses double-precision kernel values and checks convergence
explicitly. Original kernlab caches kernel columns in single precision and
defaults to a looser stopping tolerance, so exact coefficient equality is not
expected. The [SVM audit](../research/condis-svm-audit.md) records original
source pins, shared-fold numerical references, kernel bandwidths and
independently recomputed optimality checks. No native source or compiled
kernel is distributed with the Python package.

The solver updates two coefficients at a time, preserving the zero-sum
constraint and cost bounds. `solver_tolerance` defaults to 1e-8 and
`max_iterations` to 2,000 pair updates per fit; failure to satisfy the
stopping checks raises an error. Increasing the iteration limit is supported
within the work budget. The result includes per-fold KKT residuals and
duality gaps, plus final primal/dual objectives. These use the fit's response
units, which are standardized only when the native scaling rule applies.
The reported gap is the nonnegative difference between the primal objective
and the maximized dual objective, rather than a relative error.

`dual_coefficients` and `intercept` are also in the solver's response units.
For a design matrix `x` consisting of status followed by the covariates, its
stored transformation is
`z = (x / predictor_magnitude - predictor_center) / predictor_scale`.
After computing the radial kernel against the transformed training rows,
predictions are
`response_center + response_scale * (K @ dual_coefficients + intercept)`.
An unscaled fit stores identity transformations. `standardized` and
`fold_standardized` expose the full-fit and per-fold choices, and
`support_indices` identifies nonzero coefficients.

At least three observations and two training rows in every fold are required.
Zero-support-vector fits raise an error, matching native rejection of that
degenerate case. Fitting retains a kernel matrix of at most one million cells
and bounded vectors; CV has separate kernel-work and pair-update limits.
Featurewise kernel differences preserve nearby points with large common
offsets without allocating an observations-by-observations-by-features array.

## Gaussian gradient-boosting refinement

`condis_boosting_refine` adds the Gaussian `gbm` learner. Its default tuning
grid crosses 50, 100 and 150 trees with interaction settings 1, 2 and 3.
Each setting limits the number of best-first splits per tree. Shrinkage is
0.1 and each child needs at least ten sampled observations. Every iteration
samples half the training rows without replacement and fits the current
residuals, starting from the training-response mean.

The native sample-size rule requires **at least 43 training rows in every
fold**. The function rejects smaller training sets instead of silently
changing the sampling fraction or minimum node size. As with the other
refiners, the predictor matrix includes status and the supplied covariates.

```python
from mdanderson_stats import condis_boosting_refine

boosting_rows = np.arange(80)
boosting_time = 2 + (boosting_rows * 7 % 47) / 3
boosting_status = (boosting_rows % 4 != 0).astype(int)
boosting_x = np.column_stack((np.sin(boosting_rows / 3), boosting_rows % 5))
boosting_base = condis_impute(boosting_time, boosting_status)
boosted = condis_boosting_refine(
    boosting_base, boosting_x, folds=4, random_state=7,
)
assert boosted.fold_rmse.shape == (1, 4, 9)
np.testing.assert_array_equal(
    boosted.refined_time[boosting_status == 1],
    boosting_time[boosting_status == 1],
)
```

Within each fold and interaction setting, one path grows to the largest
requested tree count; shorter candidates use prefixes of that same path.
Selection minimizes mean fold RMSE, resolving equal scores by fewer trees
and then fewer splits. The selected setting is fitted afresh on the full
sample. `tree_grid` and `depth_grid` permit smaller, increasing unique grids
within 1–150 trees and 1–3 splits. `folds`, `repeats`, `fold_ids` and
`random_state` expose resampling choices. NumPy and R seeds do not produce
the same random streams.

`CondiSBoostingRefinement` retains the two grids, fold scores, their means
and sample standard deviations, selected settings, compact fitted trees,
initial mean, shrinkage, predictions and censoring diagnostics. Score columns
iterate tree count first, then interaction setting. Observed events are
restored exactly; `enforce_censoring=True` optionally clips censored results
at their observed times. Scores still assess already-imputed targets, not
held-out performance of the complete survival-analysis pipeline.

The implementation reuses sorted predictor orders and stores compact trees
with a single working prediction vector. It bounds total fitting work and
avoids a rows-by-trees prediction array during ordinary refinement.
The [boosting audit](../research/condis-gbm-audit.md) records shared-draw
construction checks against the original native kernel, including its
strict split routing and missing-branch fallback metadata. Finite numeric
covariates are required; missing-data prediction is not exposed.

## Validation and remaining coverage

Three focused tests cover hand-computed linear/step integrals and a partial
horizon, unmodified R 0.1.2 outputs with tied and unsorted inputs, preservation of
events, degenerate samples, input order and time scales from 1e-200 to 1e200.
Native adaptive integration differs from exact segment sums by approximately
1.14e-4 on the selected fixture; the reference comparison allows that integration
error. Python does not reproduce adaptive integration noise. R survival's default
near-tie handling is also not reproduced: Python groups exact ties.

A 100,000-observation example took approximately 0.033 seconds in the development
environment. The imputed times remained between observed times and maximum
follow-up. This is a local benchmark, not a cross-platform guarantee.

Three further tests cover the executed caret Gaussian model, event restoration,
censoring diagnostics and explicit clipping, and rank/scale invariance. The
comparison isolates the native final fit using exact base targets; it does not
claim reproduction of caret's full resampling pipeline.

Ridge, lasso and kNN are checked against the source-loaded caret 7.0-1 and
glmnet 4.1-10 kernels using identical input targets and held-out folds. The
comparisons cover tuning scores, selection and refitted predictions. A
collinear lasso case uses an independently checked, tighter native convergence
reference: the original default tolerance leaves measurable optimization error.
The Python fit preserves the objective without reproducing that stopping error.
Extreme time-unit checks from 1e-200 to 1e200 preserve fitted values after
rescaling; penalty grids must scale with the response units.

**Catalog status is partial.** Base CondiS and seven CondiS-X learners (linear,
ridge, lasso, kNN, neural, SVM and gradient boosting) are implemented. Random
forest refinement, remaining input/report workflows and the separately
illustrated regression prediction workflow remain pending. The
[workflow audit](../research/condis-workflow-audit.md) distinguishes the
vignette's examples from unverified interactive app behavior.
