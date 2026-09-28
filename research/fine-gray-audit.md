# Fine–Gray regression and incidence contours

The preceding goal turn made progress: stratified Cox fitting and prediction,
direct R comparisons, plotting and package verification were integrated at
`43847ca`. Catalog totals remain 62 implemented, 66 partial and 10 pending.
Entry 166, SurvivalContour, remains partial. This checkpoint targets its
Fine–Gray competing-risk model family.

Root coordinates on `feat/fine-gray`; the sole Luna worker implements on
`feat/fine-gray-luna` in its separate existing checkout. Scientific jobs are
serial with one BLAS/OpenMP thread. No dependency installation, CI redesign or
full-suite run is required. Publication remains blocked by automatic approval
review; that does not prevent local implementation and read-only source audits.

## Primary sources

The executable reference is Robert Gray's `cmprsk` 2.2-12, pinned at
[`f81411e1e3f57822796bae2a6870657e455362e9`](https://github.com/cran/cmprsk/tree/f81411e1e3f57822796bae2a6870657e455362e9).
Its DESCRIPTION declares GPL >=2, and the source credits Copyright (C) 2000
Robert Gray. Exact source files live only in ignored `research/raw/cmprsk`:

| File | Git blob |
| --- | --- |
| DESCRIPTION | `5ab1f6ef57c855b343a4e6020d77e74d5f63beed` |
| R/cmprsk.R | `759528f90b7c7706e421518d21edab69b893046e` |
| src/crr.f | `f2f473eb0e881ad0e565c1cc430521bf42e562ec` |

The unchanged 18 KB Fortran source was compiled as an arm64 shared library
with the existing compiler. R 4.4.1 loads it as `cmprsk.so`, sources the unchanged
R wrapper, and uses installed `survival` 3.6-4 for censoring Kaplan–Meier fits.
The compiler reported a deployment-version override warning; the native probe
and reference runs completed. No cmprsk package installation was needed.

The author SurvivalContour revision
[`d4645f69f23fc1146c07432f576b4c40f85e1bba`](https://github.com/YushuShi/survivalContour/tree/d4645f69f23fc1146c07432f576b4c40f85e1bba)
provides `FGContour.R` (blob `f8abb256eb13cec172cbdb480760834015ebcce1`)
and `FGContour3D.R` (blob `2be733247c328f3e13259cbe432702cefd6e94e5`).
Both evaluate `riskRegression::predictRisk` for the fitted cause, use 30 grid
points between empirical 2.5th and 97.5th percentiles by default, and fix other
numeric variables at their means or an explicit profile. Target-event times
are sorted and unique, with zero prepended if absent. The plotted quantity is
cumulative incidence, including in the 3D helper's internally named `surv`
field. The 3D helper requests no confidence bounds.

The `riskRegression` CRAN mirror was inspected at
[`08a60f7e9a24735b17c77d8b99752baaee6b6bf6`](https://github.com/cran/riskRegression/tree/08a60f7e9a24735b17c77d8b99752baaee6b6bf6)
(version 2026.03.11, GPL >=2). Source hashes are:

| File | Git blob |
| --- | --- |
| R/FGR.R | `2d2d8c167cac066c2d899fc4ad5394b4ca388ce3` |
| R/predict.FGR.R | `7390b6db42404825b583ffb876a51ff09b65ef93` |
| R/predictRisk.R | `973be560b39cef9ccbf177945148922664062976` |
| DESCRIPTION | `2431724e1680337ea8c6d0d7138a9ddc7454c7fe` |

`FGR` constructs designs then calls `cmprsk::crr`. `predictRisk.FGR` dispatches
to `predict.FGR`, which calls `cmprsk::predict.crr`, adds an initial zero row,
and selects right-continuous steps at requested times. Thus direct `crr`
numerical comparisons cover the underlying fitted model and incidence
predictions without installing the formula/UI dependency stack. Formula
encoding and the full riskRegression stack are not claimed as executed.

## Numerical contract

For target-event time u, a subject observed through u has weight one. A subject
with an earlier competing event at T retains weight G(u-)/G(T-) from their
censoring group. Earlier target events and censorings have weight zero. Target
events tied at u share one risk denominator (Breslow convention). Fixed design
columns and columns multiplied by supplied time functions enter the same model.
Censoring groups affect estimated censoring distributions, not target baselines.

The reported coefficient covariance must be the native sandwich calculation,
including its estimated-censoring correction, not inverse information. The
`crrvv` group convention accumulates q contributions in the group of each
target-event subject using competing-event residual sums from that same group;
the implementation and reference preserve that convention. No independent
theoretical correction to that native estimator is claimed here.

Baseline hazard increments are accumulated at target-event times. Prediction
with time interactions evaluates the design at those fitted event times, then
selects right-continuous cumulative steps. No new time-function evaluation is
needed between events or beyond the last event. Incidence is `-expm1(-H)`;
one minus this quantity is subdistribution survival, not all-cause survival.

## Native fixtures

`tools/reference_fine_gray.R` checks exact source MD5 hashes and generates four
CSV fixtures. Forty deterministic records include tied target, competing and
censoring times, two censoring groups, and a target event at zero. Six models
cover fixed effects with one or two censoring distributions, fixed plus time
effects, time effects alone, and selection of cause two. References retain
coefficients, score, information, inverse information, sandwich covariance,
fitted/null pseudo-log-likelihood, baseline increments, score residuals and
1,220 incidence values across mean/explicit profiles and native/custom times.

An initial native `gtol=1e-12` attempt hit line-search roundoff for the mixed
model despite a maximum score around 1.9e-10. The final harness uses `1e-10`
and asserts convergence; all five fits completed in 0.75 seconds. Its prediction
calls use one profile at a time: the native time-effects-only multiple-profile
path references an absent `cov1` argument. That wrapper defect is avoided in
the reference and need not constrain Python's matrix prediction interface.

A sixth case places target, competing and censor events together at zero.
The native R wrapper uses a relative epsilon to approximate G(t-), but this
does not move t=0 to its left. Its G(0) value can misweight earlier competing
events at later target times. To obtain a correct native reference, all times
are translated by +1 for fitting, and returned event times by -1. This fixed-
effect model is invariant to common time translation. The unshifted native
coefficients `[-0.2023311730, 0.3371024821]` differ from the correctly translated
`[-0.2039769254, 0.3359332899]`. Python must use exact left limits, including
G(0-)=1. The six-case reference generator and discrepancy probe ran in 0.79
seconds. The final fixtures contain 79 event rows and 125 scalar fit metrics.

## Python implementation and root verification

A Luna worker implemented the numerical core in a separate checkout. Root
reviewed the event-streamed likelihood, exact censoring left limits, native
censoring-group variance convention, separation check and prediction arithmetic,
then integrated the core at `856246f`. The public API now exposes the fit,
prediction and fixed-effect contour interfaces, with a usage guide at
`docs/fine-gray.md`. The GPL notice and license text are included in the package.

Fitting centers and scales covariates, evaluates risk sums on a log scale and
solves the Newton system with backtracking. A bounded separation check rejects
monotone likelihoods. Event designs are streamed; no observations-by-events-by-
columns tensor is retained. The final event-by-observation risk matrix and the
separation constraints are bounded by the work preflight. Prediction accumulates
hazards in log space and uses `-expm1(-H)`. Unscaled baseline increments may be
zero or infinity when original covariate units are unrepresentable; centered
profile predictions remain usable. Non-real inputs and unrepresentable profile
linear predictors are rejected explicitly.

The independent root adapter checked all six fits, 125 scalar metrics, 79
event rows and 1,220 incidence predictions, with warnings treated as errors.
It also checked reversed row/group order and fixed/time-covariate units scaled
by 1e-100 and 1e100. Maximum absolute differences from native references were:

| Quantity | Maximum absolute error |
| --- | ---: |
| Coefficients | 5.56e-16 |
| Sandwich covariance | 4.72e-16 |
| Observed information | 4.80e-14 |
| Baseline increments | 5.42e-16 |
| Score residuals | 4.89e-15 |
| Cumulative incidence | 6.67e-16 |

This independent run took 0.192 seconds after imports and peaked at 118.5 MiB
RSS, with zero reported process swaps. Four focused regression tests passed
in 1.54 seconds after integration, covering the native fixtures, exact time-zero
limits, a 1e8 covariate offset, monotone likelihood, work limits, complex-input
rejection and contour rejection before an oversized fit. No full suite or new
CI infrastructure was run. Ruff formatting/lint and targeted mypy passed.

## Contour contract and rendering

The native reference generator now executes the unchanged author `FGContour`
and `FGContour3D` helpers. Their render calls are captured; `predictRisk` is a
numeric adapter to the unchanged `predict.crr` with right-step selection.
Both helpers agree on grid, times and incidence for four fixed-effect models,
with mean and explicit adjustment profiles. The full formula/riskRegression
stack remains source-inspected rather than executed.

The Python contour audit matched 800 reference values across these four
models and both native/custom time grids, with maximum error 5.56e-16.
Both documented public API examples executed. The 2D and 3D plots were rendered
and visually inspected; both label the output as cumulative incidence.
This combined audit took 0.798 seconds after imports and peaked at 149.3 MiB,
with zero reported process swaps. All scientific jobs ran serially with one
BLAS/OpenMP thread; measurements describe these small reference workloads.

The right-censored Fine–Gray model family is now covered. Entry 166 remains
partial because interval-censored, parametric/spline, forest and neural families
remain open. Catalog counts stay at 62 implemented, 66 partial and 10 pending;
this distinction prevents a completed subfamily from overstating whole-entry
coverage.

## Package checkpoint

Public integration and the contour workflow are committed in `6b5b63e`.
The cached Hatch backend built both wheel and source archives without an
installation. All 467 Python modules in both artifacts matched the source;
the wheel catalog, notices and license texts matched as well. The source
archive retains the guide, audit, four reference CSVs, focused tests and R
reference harness, and excludes ignored raw downloads and compiled binaries.

An isolated interpreter loaded the wheel and executed both documented public
examples plus a 3D render. It peaked at 147.4 MiB with zero reported process
swaps. The final artifacts are rebuilt after this audit checkpoint is saved;
only static artifact checks are needed for that documentation-only rebuild.

The overall goal remains active. GitHub publication remains blocked by the
prior automatic approval review rejection, which requires unavailable approval.
No write retry or alternate publication transport was attempted.
