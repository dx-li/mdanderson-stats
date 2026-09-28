# Exact generalized-gamma survival models

The previous implementation checkpoint completed Weibull, log-normal and
log-logistic AFT models, contours and package validation at `3862e54` on local
`main`. This continuation has made progress by integrating that checkpoint and
executing independent generalized-gamma references. The full catalog goal is
active; counts remain 62 implemented, 66 partial and 10 pending.

Root coordinates on `feat/generalized-gamma`. One Luna worker reviews and then
implements in the separate existing checkout. Scientific jobs are serial with
one BLAS/OpenMP thread. No dependency installation, CI expansion or publication
retry is needed.

## Source and required models

SurvivalContour entry 166 offers both original Stacy and stable Prentice
generalized-gamma AFT models. The source is `flexsurv` 2.3.2 at
[`2aae4c8ac56823d0eac30c1a9ad654ac599b5938`](https://github.com/cran/flexsurv/tree/2aae4c8ac56823d0eac30c1a9ad654ac599b5938).
Exact source bytes and hashes are recorded in the
[preceding parametric audit](parametric-survival-audit.md) and
`tools/reference_generalized_gamma.R`. Native downloads stay under ignored
`research/raw/flexsurv` and are not redistributed in package artifacts.

Prentice parameters are location mu, positive sigma and unrestricted Q.
For nonzero Q, if G is Gamma(shape=Q^-2, rate=1), then
`log(T) = mu + sigma*log(Q^2*G)/Q`. Q=0 is exactly log-normal; Q=1 is
Weibull. The gamma CDF tail reverses for negative Q. Both signs, the exact
Q=0 case and numerically continuous behavior near zero are required.

Stacy has positive shape b, scale a and gamma shape k, with
`T = a*G^(1/b)`, G~Gamma(k,1). Its equivalent Prentice parameters are
`mu=log(a)+log(k)/b`, `sigma=1/(b*sqrt(k))`, `Q=1/sqrt(k)`.
Regression models log(a) in this original parameterization. It is the
positive-Q subset, not a replacement for unrestricted Prentice fitting.
The full joint covariance must include scale and shape estimation. Existing
ACCFLF finite-degree-of-freedom approximations do not satisfy these methods.

The native C++ density explicitly warns about cancellation near Q=0. Also,
`pgengamma.orig` in `R/GenGamma.R` computes `1-pgamma(...)` before taking a
log, which loses small upper tails. Python needs direct/logarithmic tails;
the stable native kernel supplies the reference tail values. The original
R documentation's introductory gamma transformation uses `exp(G/b)`; its
density, source and parameter transformation instead imply `G^(1/b)`.

## Executed native references

The unchanged `src/gengamma.cpp` and its six exact headers were compiled with
already installed Rcpp 1.1.1 and R 4.4.1. A Q=1 exponential-density assertion
passed. The initial compile took about five seconds; `/usr/bin/time -l` could
not query `kern.clockrate` in this sandbox, so it supplied no valid peak-memory
measurement. Subsequent reference execution reused the compiled cache.

`tools/reference_generalized_gamma.R` verifies all seven Git blob hashes and
executes those native density/CDF kernels. Its fixtures contain:

- 130 log-density, log-CDF and log-survival rows for Q=0 and both signs of
  0.001, 0.01, 0.1, 0.5, 1 and 2, over standardized values -20 through 20.
- Four exact right-censored fits from deterministic 96-row data, with positive
  and negative shape, with two covariates and with only an intercept.
- Full observed information, its inverse joint covariance, coefficients,
  log(sigma), Q and time-density log likelihood for each fitted model.
- Two positive-Q fits transformed to original Stacy coordinates
  `[log(scale) intercept, slopes, log(shape), log(k)]`, including all covariance
  cross terms and the corresponding information matrices.
- 120 prediction rows across three profiles and ten times, including zero and
  late tails.

The fit harness minimizes the native-kernel likelihood with base R `optim`
and calculates observed information with `optimHess`. It is an independent
fitting harness, **not execution of `flexsurvreg`'s full fitting/formula stack**.
All four fits converged and had positive-definite observed information;
independent five-point scores were below 1.50e-5. Fitted Q values were
0.52646833, 0.42692546, -1.3658638 and -0.80313001. The final reference run took
0.57 seconds, peaked at 94.6 MiB child RSS and reported zero child process swaps.
Python fit and prediction comparisons are recorded below.

## Distribution-kernel checkpoint

The sole Luna worker is implementing in `feat/generalized-gamma-luna`, based
on native-reference commit `dbe54e4`. Root's source review identified missing
second-order shape terms in the draft near-zero tail expansion, loss of tiny
complementary log tails, cancellation in the original Stacy density for large
k, and a slow/canceling lower-tail recovery. The worker corrected these before
the independent numerical audit.

The revised kernel uses exact normal paths at Q=0, Q and Q-squared terms in a
restricted near-zero region, stable Stirling/exponential remainders, direct
gamma tails, log-complement reconstruction and scaled tail-ratio recovery.
The original density maps to the positive-Q Prentice density to avoid subtracting
large log-gamma terms. These are exact model calculations evaluated with
float64 numerical approximations, not finite-df replacements.

Root independently checked the worker kernel against all 130 native distribution
rows with warnings as errors (relative tolerance 2e-8, absolute tolerance 1e-8).
Additional checks passed for sign-reflection identities, the exact first and
second Q derivatives at Q=0, retention of small complementary log probabilities,
and Stacy/Prentice equivalence at k=1, 4, 10,000 and 100,000,000.
The audit peaked at 114.6 MiB RSS and reported zero process swaps. Large tail-log
values can differ by 192 in absolute terms at magnitudes around 1e16 while
passing the relative criterion; the audit does not claim uniformly tiny
absolute errors in those tails.

After this kernel checkpoint, the completed root process released the
numerical lane back to Luna for fitting and prediction implementation.

## Fitting, prediction and public integration

Luna's core and four focused tests were integrated at `a8d7633`. Both models
estimate all location and shape parameters jointly and return the complete
observed information and covariance. Fitting centers/scales time and covariates,
uses bounded multistart optimization, checks the final score and positive-definite
information, and transforms the full covariance back to reported coordinates.
Prediction uses direct log tails and propagates uncertainty on log cumulative
hazard, one profile at a time rather than allocating a profile/time/parameter
tensor. Zero-time survival and its bounds are exactly one.

Root independently compared all six reference fits, all 234 scalar reference
metrics and all 120 prediction rows, with warnings treated as errors. Maximum
parameter difference was 3.30e-7, information difference 4.07e-5 and log-likelihood
difference 9.95e-13. Relative Frobenius errors of full covariance matrices were
below 1.52e-6 for Prentice and 2.84e-5 for Stacy. The largest absolute Stacy
covariance difference was 8.92e-4; the tests account for the finite-difference
Hessian and transformed native-reference coordinates. Maximum survival-probability
error was 4.46e-8. Extremely small probabilities can have log-survival differences
of order one at log-tail magnitudes of millions; the relative tolerance is 2e-6,
not a uniformly tiny absolute log-tail tolerance.

Equivalent positive-Q Prentice and Stacy fits also agreed on survival and
delta-method limits (largest bound difference 3.37e-6). The largest standard-error
difference was 6.27e-5. Covariate-unit factors of 1e-100 and 1e100, offsets of
1e8, time-unit changes of 1e100 including the likelihood Jacobian, and
uninformative zero-time censored observations passed. This audit took 3.41
seconds, peaked at 116.5 MiB RSS and reported zero process swaps.

An additional bounded probe tried the original Stacy fit on each of the two
native negative-Q datasets. Both exhausted their allowed iterations and raised
an explicit no-finite-identified-optimum error rather than reporting the
approaching log-normal boundary as a fitted finite model. This probe peaked at
112.4 MiB and took approximately 16.2 seconds total. It is evidence for these
cases, not a general proof that every nonfinite maximum is detected.

The public package exports both fit/prediction result classes and functions.
`parametric_survival_contour` accepts `gengamma` and `gengamma.orig`; it retains
aggregate allocation checks before fitting, mean/explicit profiles, default
event-time grids, selected percentile curves, and existing two-/three-dimensional
plots. Eight combinations of parameterization, profile and time-grid choice
matched core prediction exactly. Both guide examples and three rendered plot
views passed; root visually inspected the render. That audit peaked at 152.8
MiB RSS with zero process swaps.

Ten focused generalized-gamma, ordinary parametric and shared-contour tests
passed in 3.37 seconds. Scoped type checking passed for both implementation
modules, and scoped Ruff checks/formatting passed. No full suite or CI expansion
was run.

## Package checkpoint

The wheel and source distribution built successfully using the already cached
Hatchling backend, without installing anything. All 470 Python source modules
matched both artifacts byte for byte. The packaged catalog, redistribution
notices, generalized-gamma guide, audit, native-reference generator, focused
tests and four reference fixtures were also checked. Ignored native downloads
and compiled objects were absent from the source archive.

An isolated interpreter loaded the built wheel, confirmed all four new public
exports plus the shared contour function, and executed both guide examples,
including a plot, with warnings as errors. It peaked at 144.3 MiB RSS and
reported zero process swaps. Counts remain 62 implemented, 66 partial and 10
pending; entry 166 still has other model families and workflows outstanding.

The final audit text is included by rebuilding and rechecking the artifacts
after this record is saved. Those are static packaging operations; root released
the numerical lane to Luna for the spline implementation.

The next spline-model source and six executed native-kernel references are
recorded in [the spline audit](survival-spline-audit.md); its Python implementation
is in progress in the sole Luna worker's isolated checkout.

Automatic approval review previously rejected GitHub publication because it
requires unavailable approval. This does not prevent local implementation;
no publication retry or bypass was attempted.
