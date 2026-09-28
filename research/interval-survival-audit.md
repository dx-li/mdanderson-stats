# Interval-censored proportional-hazards source audit

SurvivalContour entry 166 advertises an interval-censored Cox workflow. Its
original `mets::phreg(Surv(..., type="interval2"))` route has an unresolved
response-contract mismatch documented in [remaining source leads](remaining-source-leads.md).
Counting-process partial likelihood is not an interval-censored likelihood.
The Python method is implemented against the explicit interval-censoring
likelihood and the primary `icenReg` implementation described below.

## Reference source

Clifford Anderson-Bergman's `icenReg` 2.0.16, dated 2024-01-13, is pinned at
[`26fadac37c6b54dd0e29c91c2bf07942ae120356`](https://github.com/cran/icenReg/tree/26fadac37c6b54dd0e29c91c2bf07942ae120356).
DESCRIPTION declares LGPL >=2.0 and <3. The reference script verifies each
consumed file's Git blob hash before compiling or sourcing it.

| File | Git blob |
| --- | --- |
| R/ic_sp.R | `0a51f8e92a04249ded93bd6c7476bc6b24d28275` |
| R/internal_utilities.R | `828ecb5c6afba2be725064b3a7decb505c94e15d` |
| R/user_utilities.R | `0bb26d3e9f50c57a80fbda4558367b3412e62a11` |
| src/icenReg_files/ic_sp_ch.h | `0d9cc6875f0c2af7f7b7e3ef30a84fc7ef8e9405` |
| src/icenReg_files/ic_sp_ch.cpp | `140c37b1e3a6a03dcda6157524e10fa612d54c73` |
| src/icenReg_files/ic_sp_gradDescent.cpp | `8136d221ef5988687fb6788f861136d705efb751` |
| src/icenReg_files/basicUtilities.cpp | `e153f08891dce000a0a4d1589dfbc079bc91fec6` |
| src/icenReg_files/myPAVAalgorithm.cpp | `e2e5a58d1a06502691aadeaa2caee832c8182ec3` |

## Statistical contract

The model is `S(t | x) = S0(t)^exp(x beta)`. Ordinary `(L,U]` observations
contribute `S(L | x)-S(U | x)`. Exact observations contribute the probability
mass at their singleton time. Right-censored observations contribute survival
after their censoring time. The baseline distribution has probability jumps
on Turnbull maximal-intersection intervals, including a possible right tail.
Weights multiply individual log-likelihood contributions without normalization.
Numeric covariates are centered before fitting; there is no regression
intercept separate from the baseline distribution.

Native `ic_sp` combines regression Newton updates with iterative convex
minorant and gradient updates for the baseline. `ic_sp_ch.cpp` implements the
full optimization loop; `ic_sp_ch.h` supplies the PH transformation and
derivatives. The wrapper normalizes the signed differences returned by native
`cumhaz2p_hat` to positive masses. Regression covariance is absent unless the
wrapper runs a bootstrap. A joint inverse Hessian is not silently substituted.

The baseline location within an intersection interval is not identified.
Native `getSCurves` explicitly returns the support intervals and survival
levels, describing this representational non-uniqueness. Prediction therefore
needs survival identification bounds or an explicitly chosen representative;
these bounds are not statistical confidence intervals.

The native interval wrapper shifts open lower endpoints by a fixed `1e-10`.
That is unit dependent and can erase narrow intervals. Python must preserve
endpoint order and inclusion exactly. The reference fixture retains both the
native shifted lower endpoints and their original values plus inclusion flags.

## Executed native references

`tools/reference_interval_survival.R` compiles the unchanged optimizer,
maximal-intersection builder, PH transformation and their utility functions
with installed Rcpp/Eigen. Source files are concatenated only to resolve
includes; function bodies are unchanged. Small exported wrappers call the
native entry points. The existing `adjustIntervals` wrapper supplies native
input conventions. The full R package, formula layer and bootstrap are not
installed or executed. Original sources, generated C++ and compiled objects
remain ignored under `research/raw/` and are not redistributed.

Five deterministic 96-observation cases cover mixed exact/left/interval/right
censoring, case weights, no-covariate Turnbull estimation, current-status data,
and tied exact/right-censored observations. Native convergence took 28, 30, 5,
10 and 16 iterations respectively. The fixture contains 480 input rows with
native support indices, 56 support masses, 26 fit metrics, and 183 survival
levels at support boundaries for three covariate profiles. Four current-status
support masses are zero, exercising an active baseline constraint.

The successful compile-and-reference run took 3.56 seconds, peaked at 308.5 MiB
child RSS and reported zero process swaps. It ran sequentially with one
BLAS/OpenMP thread, one compiler job and optimization disabled for the small
reference build. No dependency installation was needed. This source/reference
checkpoint established the references used for the Python implementation below.

## Python implementation and validation

`interval_survival.py` independently implements the interval likelihood using
log cumulative hazards, conditional Newton regression updates, weighted
isotonic baseline updates and likelihood backtracking. Equal adjacent baseline
values represent exact zero-mass support intervals. Common weight rescaling
and scale-first covariate standardization avoid dependence on absolute input
units. The fit exposes convergence diagnostics and raises on failed convergence.

The endpoint sweep processes exact starts, closed ends and then open starts at
each time. A right tail is retained only when the final pending start requires
it. This reproduces all 480 native observation/support indices while preserving
the original endpoint values without the native fixed perturbation.

Five native cases agree within the following absolute differences:

| Quantity | Maximum difference |
| --- | ---: |
| Regression coefficients | `1.48e-7` |
| Log likelihood | `2.01e-11` |
| Support probability masses | `9.71e-8` |
| Survival at finite support boundaries | `2.60e-7` |

Four current-status masses are exactly zero. An independent finite-difference
likelihood audit checks regression scores and ordered-baseline KKT conditions;
the largest normalized residual is `9.80e-8`. The hand-solvable intervals
`(0,1], (1,2], (0,2], (2,infinity)` reproduce masses `3/8,3/8,1/4` and both
within-interval survival bounds.

The root integration audit checks covariate units `1e-100` and `1e100`, time
units `1e-100` and `1e100`, and global weight factors `1e-200` and `1e200`.
Both public guide examples ran successfully. Four mean/explicit-profile and
default/custom-time contours, including their percentile curves, match direct
predictions; four 2D/3D bound views were rendered and visually inspected.
The warnings-as-errors audit took 1.35 seconds and peaked at 160.6 MiB RSS with
zero reported process swaps, using one numerical thread.

Three focused regressions cover native fits, endpoint assignments, masses and
curves, the hand-solvable case and time units, and weight/input behavior. No
full-suite or simulation campaign was run for this addition. The API provides
identification bounds, not bootstrap or analytic confidence limits. Stratified
interval fits, interval competing risks and other outstanding SurvivalContour
families remain pending; catalog entry 166 remains partial.

The integrated three-test run passed in 2.06 seconds with warnings treated as
errors; Ruff formatting/lint and targeted mypy checks passed. Wheel and source
archives were built from cached build dependencies without installing packages.
All 475 Python modules were byte-compared against both archives, and all eight
new public exports and both guide examples passed against the wheel in an
isolated interpreter (144.2 MiB peak RSS, zero reported swaps).
