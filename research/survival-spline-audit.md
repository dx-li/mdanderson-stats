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

Python implementation and validation are in progress in the sole Luna worker's
separate checkout. This reference checkpoint does not claim a completed Python
spline workflow or change any catalog status.
