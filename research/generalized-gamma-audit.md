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
The fit and prediction comparisons against Python remain pending.

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

This is a kernel checkpoint in the worker checkout. Fitting, full covariance,
prediction, public integration and package checks remain in progress; no new
catalog entry or complete generalized-gamma workflow is claimed. After this
audit the completed root process released the numerical lane back to Luna.

The next spline-model source has also been pinned and inspected; its verified
files and implementation requirements are in
[remaining source leads](remaining-source-leads.md). No spline implementation
or numerical execution is claimed.

Automatic approval review previously rejected GitHub publication because it
requires unavailable approval. This does not prevent local implementation;
no publication retry or bypass was attempted.
