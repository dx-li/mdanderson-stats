# CondiS-X learner source and numerical audit

Catalog entry 157 uses CondiS 0.1.2. Its base imputation and Gaussian linear
refinement were previously implemented. This checkpoint covers the source
contracts and native references for ridge, lasso and nearest-neighbor
refinement; it does not complete the other CondiS-X learners or the application.

## Pinned primary sources

The CondiS archive and caret model metadata are recorded in
[`docs/condis-sources.json`](../docs/condis-sources.json). The additional source
files are pinned to caret 7.0-1 at
[`31a3ab0a746ca84c564f9f5cc0214d52570c5a04`](https://github.com/cran/caret/tree/31a3ab0a746ca84c564f9f5cc0214d52570c5a04)
and glmnet 4.1-10 at
[`2b85f4d60b7629929ebd8f9c87b2179b721e50b5`](https://github.com/cran/glmnet/tree/2b85f4d60b7629929ebd8f9c87b2179b721e50b5).
Every saved file has a verified Git blob hash and a SHA-256 digest in that JSON.
The original source remains under ignored `research/raw/CondiS` and is not
included in the Python distribution. These pins define the compatibility
target; the 2022 publication does not pin its original dependency versions.

## Common workflow

`CondiS-X.R` fits `pred_time ~ .` to the imputed time, censoring-status column
and supplied covariates. The wrapper computes a centered/scaled covariate
object, but never passes it to `train`. Therefore kNN uses raw predictors;
glmnet performs its own standardization. Status is a predictor in both cases.
Full-sample fitted predictions are returned after restoring observed events.

The native `trainControl(method="repeatedcv")` defaults to ten folds and
**one** repeat. Numerical targets are stratified using at most four intervals
between empirical quantiles: the number of break probabilities is
`max(2, min(5, floor(n/k)))`. Folds are balanced within each nonempty stratum,
then randomly shuffled. Native and Python random streams must not be described
as identical. Explicit held-out fold assignments permit numerical comparisons.

Regression tuning minimizes the unweighted mean of fold RMSEs, rather than
pooling all squared residuals. caret sorts models before selecting the first
minimum: larger lambda first for ridge/lasso, larger k first for kNN.
The selected setting is refitted on the entire supplied sample. The numerical
reference preserves that ordering and full-sample refitting.

These tuning errors concern the supplied imputed targets. The native workflow
constructs those targets before learner cross-validation. They do not estimate
held-out survival prediction accuracy for the complete imputation pipeline.

## Ridge and lasso

The native tuning grid contains ten linearly spaced lambdas from 0.01 to 10,
with alpha fixed at zero for ridge or one for lasso. caret's model metadata
fits one default glmnet path per alpha; the tuning lambdas are **not** passed
as the path's lambda argument. Predictions interpolate coefficients linearly
between path points and clamp to the path endpoints, with `exact=FALSE`.
The selected final model is another default path fit on all observations.

The Gaussian driver centers predictors and response and uses population
standard deviations (divisor n). Constant predictors are excluded. Let Z be
the standardized predictor matrix and r the standardized response. At internal
penalty t, it solves

`||r - Z b||^2 / (2 n) + t * ((1-alpha) ||b||^2 / 2 + alpha ||b||_1)`.

The reported lambda is `t * sd_pop(y)`; original-unit slopes are
`b * sd_pop(y) / sd_pop(x)`. This response scaling is material for ridge.
The default lambda grid therefore depends on the units of time; multiplying
the requested grid along with time is necessary for a unit-invariance check.

The default path has at most 100 points. Its geometric minimum ratio is 0.01
when observations are fewer than original predictor columns, otherwise 0.0001.
The first fit uses a very large penalty; its reported lambda is then replaced
by the extrapolated sequence endpoint. The second internal lambda is the
largest absolute initial standardized gradient divided by
`max(alpha, 0.001)`, multiplied by the geometric ratio. From five points onward,
the path can stop when the proportional explained-deviance gain is below 1e-5
or explained deviance exceeds 0.999. A fixed-grid ridge/lasso solve alone does
not reproduce this prediction contract.

## Nearest-neighbor regression

The default caret grid is k=5,7,9. Distances use raw status and covariate
columns, without predictor-by-predictor scaling. Predictions average the
retained nearest neighbors, including ties, and k is clipped to training size
with a native warning when necessary. Full-sample predictions include the
observation itself.

`src/caret.c` has a subtle retained-distance rule. Its candidate and final
extra-neighbor checks use a relative tolerance of 1e-4 on squared distances,
but its insertion buffer grows only on exact ties to the current kth distance.
Consequently, including every distance within 1e-4 of the final cutoff is not
equivalent. The reference contains exact ties, near ties and reordered rows to
make this distinction observable. The original fixed-size tie buffer is a
native implementation limit, not a mathematical kNN requirement.

## Reference harness

`tools/reference_condis_models.R` verifies source hashes before execution. It
builds only the native dense Gaussian wrapper with its unchanged Gaussian
headers and factory constants, using installed Rcpp/RcppEigen. It sources
the original glmnet R fitting, coefficient extraction and prediction functions.
A read-only control shim supplies the factory fields read by `glmnet.R`;
progress display is disabled. No fitting formula or numerical solver is patched.

The same harness builds caret's unchanged small C kernel and sources its
nearest-neighbor and fold functions. The serialized caret learner fit function
is executed with only `glmnet::glmnet` dispatch redirected to the source-loaded
function. The resampling loop aggregates the resulting fold predictions using
the verified caret rule. This is a source-only reference, not execution of the
whole installed caret training stack.

Four bounded inputs cover an ordinary sample, two changes of time units and
more predictors than observations. Fixtures retain explicit native fold IDs,
per-fold and mean tuning errors, selected parameters, every full-sample grid
prediction, final refined times and complete fitted Gaussian paths.
Input imputed targets come from the already validated exact-segment Python
base method; these references isolate learner behavior from native adaptive
integration noise.

The reference run completed on 2026-09-28 with R 4.4.1, in 5.34 seconds using
303.2 MiB peak resident memory and zero swaps. The Gaussian-only compilation
and the small C compilation ran sequentially with optimization disabled and
one build job. No R package was installed. Python learner agreement is checked
separately; successful reference generation alone does not prove a port correct.
