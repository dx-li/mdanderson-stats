# CondiS-X radial SVM source and numerical audit

Catalog entry 157 remains partial. This audit records the original SVM
workflow and its executed numerical references; reference generation alone
does not establish correctness of the Python learner.

## Source contract

The original CondiS 0.1.2 `CondiS-X.R` calls caret `svmRadial` with
`pred_time ~ .`, including censoring status as an ordinary predictor. Its
preprocessing object is not used. Observed event times are restored after
full-sample prediction, as for the previously implemented learners.

The pinned caret model metadata generates one bandwidth and three costs:

- `sigest(as.matrix(x), scaled=TRUE)` estimates three bandwidths. The tuning
  bandwidth is the mean of the first and third, not the median estimate.
- Costs are `2^((1:3)-3)`, or 0.25, 0.5 and 1. The model sorts by increasing
  cost, then decreasing sigma. The first minimum mean fold RMSE wins.
- The chosen setting is refitted to all supplied rows. Resampling uses the
  existing repeated-CV contract: ten folds, one repeat by default, numeric
  target strata and unweighted means of held-out fold RMSEs.

`sigest` samples two vectors of `floor(n/2)` row indices independently **with
replacement**. After predictor handling below, it computes squared distances
between the paired rows, excludes exactly zero distances and takes reciprocals
of their 0.9, 0.5 and 0.1 quantiles. The final sigma is the mean of the first
and last reciprocal. This is one full-sample bandwidth estimate before CV;
individual training folds do not re-estimate it. Explicit pairs are saved in
the references. Equal NumPy and R seeds do not imply equal pairs or folds.

The backend is kernlab 0.9-33 at
[`d8f05c9b1b8d220deb98fdf28cd33471f17d5eae`](https://github.com/cran/kernlab/tree/d8f05c9b1b8d220deb98fdf28cd33471f17d5eae).
[`docs/condis-svm-sources.json`](../docs/condis-svm-sources.json) records 22
verified original files with Git blob hashes, SHA-256 digests and sizes.
The original files stay under ignored `research/raw/CondiS/future-learners`.
This is an explicit compatibility target, not a claim that the 2022
publication pinned that later dependency version.

## Scaling and model

The matrix method in `R/ksvm.R` defaults to epsilon-SVR, with epsilon=0.1,
solver tolerance=0.001 and shrinking enabled. It standardizes each predictor
and the numeric response using training means and sample standard deviations
(divisor n-1). Prediction uses those same training transformations.

There is a material native exception: **any constant predictor disables all
predictor scaling and response scaling** for that fit. `sigest` independently
uses the same all-or-nothing predictor-scaling rule. A constant status column
can therefore trigger the exception, even if all other covariates vary. The
native warning is retained in the reference metadata. Silently dropping the
constant column or scaling only the remaining columns would change the model.

With transformed inputs z, the radial kernel is

`K(i,j) = exp(-sigma * ||z_i-z_j||^2)`.

The dual minimizes

`0.5 beta' K beta + epsilon sum(abs(beta)) - y' beta`,

subject to `sum(beta)=0` and `-C <= beta_i <= C`. Predictions in training
response units are `K beta - rho`; native rho has the opposite sign from an
additive intercept. Original response units are restored only when training
response scaling was applied.

For residual e=y-(K beta-rho), free positive coefficients require e=epsilon,
free negative coefficients require e=-epsilon, zero coefficients require
abs(e)<=epsilon, positive bound coefficients require e>=epsilon and negative
bound coefficients require e<=-epsilon. The primal objective is

`0.5 beta' K beta + C sum(max(abs(e)-epsilon, 0))`.

The primal objective plus the minimized dual objective is the duality gap.
The reference also records KKT, equality and box residuals. With no free
support vector, native SMO uses the midpoint of its feasible rho interval.
The fixed all-bound example saves beta=(-0.01,-0.01,0.01,0.01) and rho=-3,
with a gap at floating-point rounding error.

The original C++ cache stores kernel columns as **float32**, although
coefficients and other calculations use double precision. Native default
stopping error and cache rounding are distinct from mathematical model error.
Native `ksvm` rejects a zero-support-vector solution. A well-defined Python
extension for such degenerate data must be documented explicitly.

## Executed reference

`tools/reference_condis_svm.R` verifies every source hash, compiles unchanged
`svm.cpp` and its numerical link dependencies, then sources the original
S4 class definitions, kernels, matrix fit/predict methods and `sigest`.
Only caret's namespace dispatch is redirected to these source-loaded
functions. The numerical kernel and its fitting/prediction formulas are
unchanged. This executes native `ksvm` and `predict`, but the surrounding
fold aggregation is a source-verified loop rather than the complete installed
caret training stack. No R package is installed.

Three inputs cover ordinary data, more predictors than observations, and a
constant predictor. Each saves explicit bandwidth pairs and folds, all three
costs, per-fold/mean scores, selection, full-sample fitted values, coefficients,
rho, scaling and convergence certificates. Base imputed targets are reused
from the separately validated exact-segment CondiS method. The comparisons
isolate the learner from native adaptive-integration differences.

The references run both native default tolerance 0.001 and tightened
tolerance 1e-8. All three cases select C=1. Their sigmas are approximately
0.241921756, 0.0245045965 and 0.339662424, respectively. Maximum full-fit KKT
violations at default tolerance are 5.1e-4 to 7.1e-4; after tightening they
are 5.2e-8 to 8.3e-8. The remaining tight-reference duality gaps are about
1.1e-7 to 1.4e-7, consistent with the float32 kernel cache.

The initial compile/reference run completed in 4.33 seconds using 162.5 MiB
peak child resident memory and zero swaps. The complete reference with
certificates and the all-bound case ran in 2.21 seconds using 152.7 MiB and
zero swaps with the compiled kernel cached. Both used one build job and one
BLAS/OpenMP thread.

An independent NumPy calculation reproduced the sampled bandwidths within
3.47e-18, all full-sample predictions within 7.11e-15 and duality-gap
calculations within 5.69e-14. Kernel matrices were positive semidefinite to
rounding accuracy; coefficient sums and cost bounds were checked directly.
These verify the saved reference interpretation. Python solver agreement is
checked separately before coverage is claimed.
