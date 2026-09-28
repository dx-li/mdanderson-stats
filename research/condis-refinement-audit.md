# CondiS-X learner source and numerical audit

Catalog entry 157 uses CondiS 0.1.2. Its base imputation and Gaussian linear
refinement were previously implemented. This checkpoint covers the source
contracts, native references and Python tuning/refit workflows for ridge,
lasso and nearest-neighbor refinement; it does not complete the other CondiS-X
learners or the application.

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

The ordinary sample selects ridge lambda 0.01, lasso lambda 0.01 and k=7;
the wide sample selects ridge lambda 10, lasso lambda 0.01 and k=5. The
ordinary Gaussian paths contain 100 ridge points and 62 lasso points; the
wide paths each contain 100 points. In the wide ridge fit, the smallest fitted
lambda is approximately 25.45, so every requested value in 0.01..10 clamps to
the same full-sample path endpoint. This fixture exercises endpoint behavior
and tuning ties rather than only checking a fixed-penalty solution.

The native k=1 prediction at zero for training coordinates
`[1, 1, sqrt(0.99999)]` and responses `[10, 20, 30]` is 20. Putting the closest
coordinate first, with coordinates `[sqrt(0.99999), 1, 1]` and those same
responses, yields 10. A simple sort followed by an inclusive distance-tolerance
mask would miss this insertion-order distinction.

## Lasso convergence reference

The wide-data lasso fixture exposes the effect of glmnet's default coordinate
stopping threshold. `tools/reference_condis_lasso_convergence.R` repeats the
unchanged native Gaussian kernel with `thresh=1e-15`, fixing the penalty path
to exactly the original reported lambda values. It checks the full sample and
each original held-out fold, and saves all candidate predictions and tuning
scores in `tests/fixtures/condis-lasso-converged.json`.

For the standardized objective defined above, the maximum KKT violation is
`|Z_j' residual/n - t sign(b_j)|` on nonzero coefficients and
`max(|Z_j' residual/n|-t, 0)` on zero coefficients. At the final full-sample
path point, the native default violation is 2.8601e-4 and the tighter native
violation is 2.8774e-8. The objective decreases from 0.017867233952541028 to
0.017865457505021089. These certificates independently identify default-native
stopping error under collinearity; matching that error is not a correctness
requirement for the Python implementation.

The tighter reference, including the original reference setup, completed in
6.57 seconds with 300.3 MiB peak resident memory and zero swaps. It installs
nothing and does not change a fitting formula, penalty path or random fold
assignment. The original default-tolerance references are retained separately.

The integrated public-API comparison also isolated smaller stopping error in
fold 8 of the ordinary sample: at requested lambda 0.01, default-native RMSE
is 2.413015715534771, tighter-native RMSE is 2.4130258325643044 and Python
RMSE is 2.413025827506931. Across all ordinary folds and requested lambdas,
Python and the tighter native reference agree within 2.10e-9 elementwise
relative error. The full-sample fitted values agree within 3.59e-11.
The affected fold's endpoint KKT violation drops from 2.58e-6 to 2.15e-12
under the tighter native tolerance, with objective decreasing from
0.2578008581822215 to 0.2578008581774756.

Running `tools/reference_condis_lasso_convergence.R ordinary` saves this
additional reference in `tests/fixtures/condis-lasso-ordinary-converged.json`.
It completed in 5.54 seconds using 303.6 MiB peak resident memory and zero
swaps. Both ordinary and wide lasso regression checks use tightened references;
the original default references remain available to diagnose compatibility
differences rather than making Python reproduce their stopping errors.

## Python implementation checks

`condis_regularized_refine` exposes the three learners through one tuning/refit
interface and returns each held-out fold score, mean and sample-SD RMSEs,
selected setting, full-sample predictions and event-restored refined times.
The implementation uses bounded NumPy operations without parallel processes.

The implementation worker compared every grid prediction and fold score with
the native fixtures. Its relative errors below divide the maximum absolute
difference by the maximum absolute reference value over the whole prediction
or fold-score matrix; they are not maxima of elementwise relative errors.
kNN differences were approximately 1e-15. Ridge fitted
surfaces agreed within 9e-9 relative and fold scores within 1.2e-7; the ordinary
lasso surface agreed within 1.1e-8 and its fold scores within 1.6e-6. Against
the tighter wide-data lasso reference, surface differences were at most
1.75e-7 relative and fold-score differences at most 8.66e-7. Its full-sample
endpoint objective was approximately 0.01786546 with KKT violation 2.88e-8,
consistent with the independently tightened native solver.

Changing time units by 1e-200 and 1e200, scaling requested lambdas with those
units, preserved fitted predictions after rescaling within 2.64e-16 relative
in the worker's bounded check. Constant responses and entirely constant
designs produced finite intercept-only mean predictions. kNN averaged retained
ties. These degenerate regularized fits are deliberate Python extensions.

The compact persisted regression compares ordinary shared-fold scores,
selection and final predictions for all three learners and the tighter
wide-data lasso reference. Existing behavioral checks cover event restoration,
explicit folds, seeded reproducibility and native kNN tie insertion. Reference
generation and full native solvers are not part of ordinary CI.

The integrated base, linear and regularized CondiS checks passed all 18 tests
in 3.66 seconds (3.99 seconds including process setup), with 136.8 MiB peak
child resident memory and zero swaps. Targeted Ruff lint/format and mypy
checks also passed. Validation ran with one BLAS/OpenMP thread. The full
repository test suite was not rerun for this focused learner addition.

## Source leads for the remaining learners

[`condis-remaining-sources.json`](condis-remaining-sources.json) records eight
verified primary files for nnet 7.3-21, randomForest 4.7-1.2, kernlab 0.9-33 and
gbm 2.3.1. These are research leads, not implemented coverage or assertions
that current package defaults match every dependency version used in 2022.
Their source snapshots remain ignored under `research/raw/CondiS/future-learners`.

CondiS-X calls caret's `nnet` regression with a linear output; its extra range
preprocessing object is also unused. The nnet wrapper exposes a single hidden
layer, random initial weights, weight decay and an iteration limit. Verify
the caret size/decay grid, initialization and resampling/refit seed schedule
before implementing that remaining learner.

The random-forest branch supplies `mtry=sqrt(ncol(covariates))`, even though
status is included in the actual design matrix. This is not the backend's
default one-third-of-predictors rule. The native regression backend defaults
to 500 trees, bootstrap sampling with replacement and terminal node size five;
its handling of the noninteger supplied mtry must be preserved explicitly.

The SVM branch calls caret's `svmRadial`, while gradient boosting uses `gbm`.
Their backend scaling and objective definitions, caret-generated tuning grids,
training randomness and final refit behavior require separate native checks.
Existing survival forests are not a substitute for regression forests, and the
three learners in the current checkpoint do not substitute for these four.

### Neural-learner source continuation

[`condis-nnet-sources.json`](condis-nnet-sources.json) additionally pins nnet
7.3-19 at `4600c58673b3e4ba829286bf61b2b6f5ecf5da56`, matching the nnet
version already available in the local R installation. Its original R wrapper,
DESCRIPTION and C implementation are saved and hash-verified. Source retrieval
does not itself establish fit parity. The executed reference below uses the
already-installed native package; Python neural-learner coverage remains pending.

The executed caret metadata uses hidden sizes 1, 3 and 5 crossed with decays
0, 0.1 and 0.0001. It sorts by increasing size and decreasing decay before
selection. `CondiS-X.R` passes `linout=1`; its range preprocessing remains
unused. The C source confirms **logistic hidden units**, not tanh, and a linear
output. The native sigmoid is set exactly to zero below -15 and one above 15.
For this regression, the objective is the sum of squared errors plus decay
times the sum of squared weights, including bias weights; neither term is
divided by the number of rows. Native initial weights are uniform on
[-0.7,0.7], with default maximum 1,000 weights and 100 BFGS iterations.

The C optimizer delegates to R's `vmmin`, with absolute tolerance 1e-4 and
relative tolerance 1e-8. A faithful next implementation needs explicit initial
weights for cross-language numerical comparisons and must report its optimizer
status; different random streams or an alternative optimizer can lead to
different local solutions. Merely fitting a generic neural network would not
reproduce this contract.

The same provenance file pins R's `src/appl/optim.c` at SVN revision 85143
through its Git mirror. This is a source lead predating R 4.4.1, not a verified
complete snapshot of that release. `vmmin` uses an identity inverse-Hessian
start, Armijo acceptance coefficient 1e-4, step reduction by 0.2, curvature
updates and restarts, and its specific absolute/relative objective stopping
rules. These details are available for the next implementation; equivalence
must still be checked against the installed native optimizer.

### Executed neural references

`tools/reference_condis_nnet.R` executes the original caret fit wrapper with
the installed nnet 7.3-19 and R 4.4.1. It verifies the three pinned nnet source
files and the caret model snapshot first. It does not install packages or run
the complete caret training pipeline: it uses the verified grid, explicit
shared fold assignments and unweighted mean fold RMSE aggregation.

`tests/fixtures/condis-nnet-native.json` retains 117 fold fits, 18 full-sample
fits and 12 optimizer trajectory fits across the ordinary and wide inputs.
Each fit has explicit deterministic starting weights, final weights, objective
value, predictions and native convergence status. These starts isolate
optimizer behavior from differing R and NumPy random streams; they are not a
claim to reproduce a native default seed. The trajectory uses the same start
at iteration limits 0, 1, 2, 5, 10 and 100. Initial gradients come directly from
nnet's registered native derivative routine after setting up the native
network and retaining its training buffers.

An additional four-row example has two binary predictors, one hidden unit,
initial weights `(0, log(2), -log(2), 1, 2)` and response `(1, 2, 3, 4)`.
Its exact hidden outputs are `(1/2, 2/3, 1/3, 1/2)` and predictions are
`(2, 7/3, 5/3, 2)`. The squared-error sum is `62/9`; decay 0.1 adds
`0.1 * (5 + 2 * log(2)**2)`. Its analytical gradient, including bias decay,
matches the original kernel within 1e-13 absolute. This checks the weight
ordering and objective independently of optimizer behavior.

The native reference generation passed in 0.79 seconds with 86.9 MiB peak
child resident memory and zero swaps, using one process and one BLAS/OpenMP
thread. Independent NumPy reconstruction of all retained fits found maximum
absolute differences of 1.43e-14 in predictions, 4.55e-13 in objectives,
1.14e-13 in initial gradients and 3.34e-15 in fold RMSEs. Centered finite
differences of the initial objectives agreed with the analytic gradients
within 5.26e-8 after dividing each gradient difference by
`max(1, abs(gradient))`. The analytical four-row case also passed.

Both reference cases select size 5 and decay 0.1, with mean fold RMSEs
1.7287934888481147 (ordinary) and 2.9362319980942626 (wide). At the native
default iteration limit of 100, **55 of 90 ordinary and 17 of 27 wide fold
fits have convergence code 1**, as do both selected full-sample fits. These
are valid native compatibility observations, not evidence that their neural
optimization converged. A Python implementation must expose this status and
allow a larger iteration limit. With `maxit = 0`, native code returns status
0 even though optimization was not attempted; that case is retained solely
as an initial-objective reference and must not be described as convergence.

These checks validate the saved native reference data and the neural
objective/gradient contract. No Python optimizer, tuning interface or
neural-learner completion is claimed at this checkpoint. Native generation
is a research tool and is not added to ordinary CI.

The fixture also retains 18 checkpoints of an actual size-5, decay-0.1 CV
fit: the first fold of each case, at iteration limits 0, 1, 2, 5, 10, 20,
40, 60 and 100. `cv_trajectory` records the training/test row indices,
explicit start, fitted weights, held-out predictions, objective, status and
native gradient at each checkpoint. These let a port distinguish incorrect
fold/start selection or optimizer branches from differences that emerge late
in a nonconvex fitting trajectory. They do not assert identical solutions
across different floating-point implementations.

Regeneration with these checkpoints took 0.77 seconds and 86.1 MiB peak child
resident memory, with zero swaps. All previously saved reference results
were unchanged. Independent reconstruction of the new checkpoint objectives,
gradients and held-out predictions agreed within 2.28e-13, 1.21e-13 and
7.11e-15 absolute, respectively. Each final checkpoint exactly reproduces
the corresponding saved native CV fit.

`cv_trajectory_fold_one` adds the same nine limits for zero-based fold 1 of
each case, including the ordinary fit where an initial Python optimizer
draft diverged most. These 18 additional checkpoints reproduce their saved
native final fits exactly. Independent objective, gradient and prediction
reconstruction agrees within 9.10e-13, 2.28e-13 and 3.56e-15 absolute. All
earlier fixture results remain unchanged. Comparison of a Python optimizer's
path with these checkpoints is a separate validation step.

### Python neural fitting and numerical limits

`condis_neural.py` implements the logistic-hidden/linear-output network,
summed-error objective, bias-inclusive decay and inverse-Hessian BFGS updates
in NumPy. It follows the native line-search, restart and stopping logic,
including the grouped Hessian update expression. The nine-candidate tuning
interface accepts explicit fold assignments and per-fit starting weights,
then refits the selected candidate. Results retain fold objectives, iteration
counts and stopping codes. A readable single-fit status distinguishes
`not_run`, `iteration_limit` and `stopped`; the latter is not a certificate
of first-order or global optimality. The returned objective is recomputed at
the returned weights, and nonrepresentable objectives, gradients or search
directions raise an error.

Comparisons use exactly the saved targets and starts. At the difficult
ordinary fold-1 size-5/decay-0.1 trajectory, objectives and gradients evaluated
at native checkpoint weights agree within floating-point error, and fitted
weights agree to approximately 5e-7 through iteration 40. Later iterations
separate substantially: the native objective is about 71.36 and the Python
objective about 95.83 at iteration 100, with both reporting the iteration
limit. The wide fold-1 trajectory remains close through iteration 100, with
an objective difference around 1e-8. These observations support the model and
early optimizer arithmetic, while demonstrating that final nonconvex fitting
paths are sensitive to floating-point evaluation order.

Both full tuning comparisons select native candidate index 6 (size 5,
decay 0.1), and both selected refits report the native iteration-limit status.
The ordinary selected full-data objective is about 123.3763 in Python versus
122.9493 natively; the wide selected objective agrees within 9e-12. Across
all CV candidates, maximum fold-RMSE differences are approximately 1.663
(ordinary) and 0.544 (wide), and maximum mean-score differences are 0.160 and
0.276. Matching selected candidates on these two fixtures does not establish
general selection parity. The package implements the native statistical
model and tuning choices; it does not promise identical local optima or
final CV scores across R and NumPy. Increasing the iteration limit allows
more optimization but does not guarantee a global optimum.

After integration, the 21 focused base, linear, regularized and neural CondiS
tests passed in 3.58 seconds (138.6 MiB peak child resident memory, zero swaps).
The neural checks cover a closed-form objective/gradient, actual optimization,
native iteration-10 weights and objective, the nine-candidate workflow and
event restoration. The iteration-10 objective comparison uses an explicit
absolute tolerance with zero relative tolerance. Targeted Ruff, formatting
and mypy checks passed. Four implemented-learner guide examples, including
neural tuning/refitting, executed successfully. No broad repository test suite
or additional CI workflow was added for this learner.
