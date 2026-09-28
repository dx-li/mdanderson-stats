# SurvivalContour spline models

Entry 166 remains partial. Its spline family uses Royston–Parmar models from
Christopher Jackson and contributors' GPL >=2 `flexsurv` 2.3.2, pinned at
[`2aae4c8ac56823d0eac30c1a9ad654ac599b5938`](https://github.com/cran/flexsurv/tree/2aae4c8ac56823d0eac30c1a9ad654ac599b5938).
The native source and required behavior are recorded in
[remaining source leads](remaining-source-leads.md).

## Native reference checkpoint

`tools/reference_survival_spline.R` verifies exact Git blob hashes for
`src/splines.cpp`, `R/spline.R`, `R/deriv.R` and `R/deriv2.R`. It compiles the
unchanged C++ basis functions using installed Rcpp 1.1.1 and sources the
unchanged R distribution functions. Original sources and compiled objects
remain under ignored `research/raw/` and are not redistributed.

The generated fixtures contain:

- 88 basis/derivative values with zero, one or four internal knots, including
  both linear tails and knot locations.
- 72 distribution rows for hazard, odds and normal scales, with native ordinary
  probabilities and densities plus direct log-domain equivalents.
- One deterministic 96-row right-censored dataset with two covariates.
- Six fitted models: all three scales with zero or four internal knots.
- 24 knot coordinates, 522 parameter/information/covariance/likelihood values,
  and 162 prediction rows across three profiles and nine positive times.

Fitting uses base R `optim` with the native density/survival kernels; this is
an independent fitting harness, not execution of the full `flexsurvspline`
formula and fitting stack. All six fits converged with positive-definite
observed information. Independent five-point scores were below 3.25e-5.

The derivative of the transformed survival function must stay positive
between observations as well as at observed event times. The harness checks
its global minimum by inspecting the quadratic derivative on every knot
interval, including interior vertices, and both constant tails. All six native
fits pass; their minimum slopes range from 1.0918 to 2.3385. These references
therefore describe proper monotone survival curves over the full time support.

Initial compilation and reference execution took 4.97 seconds and peaked at
194.5 MiB child RSS. The final run, including score and global monotonicity
checks, reused the compiled cache, took 1.68 seconds and peaked at 137.4 MiB.
Both runs reported zero child process swaps. Scientific jobs ran sequentially
with one BLAS/OpenMP thread, without installing dependencies.

## Python implementation and checks

`survival_spline.py` implements all three links with either zero through 32
internal knots or explicit internal log-time knots. The default is four.
Fits normalize log time and covariates, use analytic scores and observed
information, and require a finite stationary point, positive-definite
information and a globally positive link derivative. Predictions use exact
linear tails and bounded allocations. Joint covariance includes baseline/slope
cross terms and is transformed back to the reported parameter coordinates.

Across the six native fits, maximum coefficient differences are 9.83e-6,
log-likelihood differences are below 3e-12, and ordinary survival probability
differences are below 5.8e-8. The largest relative log-survival difference is
1.01e-5. Absolute errors in enormous negative log survival are not probability
errors: the hazard reference reaches log survival below -5.5 billion at time
1e6, where both ordinary probabilities underflow to zero.

The maximum joint-covariance relative Frobenius difference is 8.58e-4 against
the native harness's raw-coordinate `optimHess` inverse. That finite-difference
reference is less accurate for the highly correlated four-knot coefficients.
An independent five-point derivative audit in normalized coordinates checked
all six models away from their optima: score errors were below 1.19e-8,
information relative Frobenius errors below 6.65e-12, and inverse-information
differences below 2.27e-8. This audit peaked at 112.8 MiB RSS. It checks
derivatives of the implemented likelihood rather than claiming an additional
native fitting reference.

Zero-knot fits, likelihoods, complete covariance matrices and predictions were
also compared with the independent Weibull, log-logistic and log-normal AFT
implementations after the full parameter/covariance transformation. Predictions
included time zero and times from 1e-30 through 1e10. Four-knot fits retained
their predictions under covariate-unit changes of 1e-100 and 1e100, offsets of
1e8, and time-unit changes of 1e100. The likelihood's time-unit Jacobian,
explicit/default-knot equivalence and zero-time censor invariance also passed.
The fit audit took 0.39 seconds and peaked at 116.4 MiB; probability/confidence
limit changes from covariate units and offsets stayed below 4e-9.

The public contour dispatcher supports `spline_hazard`, `spline_odds` and
`spline_normal`, including `spline_k` and `spline_internal_knots`. Twelve
mean/explicit-profile and default/custom-time surfaces matched direct model
predictions, with matching percentile profiles and curves. Both public guide
examples executed, and three existing plot views were rendered and inspected.
The contour/guide/plot audit peaked at 153.0 MiB RSS. All reported numerical
audits used one BLAS/OpenMP thread and reported zero process swaps.

Twelve focused spline, generalized-gamma, AFT and shared-contour tests passed
in 2.43 seconds. Ruff checks and focused mypy checks passed. The wheel and
source distribution were built without installing dependencies; all 471 Python
modules were byte-matched to source, alongside catalog/notices and the spline
guide/reference artifacts. Both guide examples also ran in an isolated
interpreter against the wheel (132.0 MiB peak RSS, zero reported swaps).
No full-suite run or CI expansion was performed for this checkpoint.

## Remaining scope

This supports unweighted exact and right-censored data, static numeric
covariates on the link intercept, and the native `rp` basis. Left/interval
censoring, delayed entry, ancillary/time-varying covariate effects, alternate
bases and native simulated-parameter confidence limits are not implemented by
this interface. Python reports deterministic delta-method pointwise limits
using the full joint covariance. The full native app/formula stack is not
reproduced. Entry 166 remains partial, with forests, neural models and
interval-censored workflows among its remaining families.
