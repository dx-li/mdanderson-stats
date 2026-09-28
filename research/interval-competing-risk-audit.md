# Interval-censored competing-risk implementation contract

SurvivalContour's `FGIntContour.R` calls `intccr::predict.ciregic` for the
first of two competing causes. This is a distinct interval likelihood, not a
wrapper around a right-censored Fine–Gray fit. Implementation remains pending;
the notes below record inspected primary source and executed R references,
not a validated Python model.

## Pinned source

The author helper is pinned at
[`YushuShi/survivalContour`, `d4645f69f23fc1146c07432f576b4c40f85e1bba`](https://github.com/YushuShi/survivalContour/tree/d4645f69f23fc1146c07432f576b4c40f85e1bba),
with helper blob `5dea1bd0dcc52f31162616e08a70183120851cc0`.
The numerical package is Giorgos Bakoyannis and Jun Park's `intccr` 3.0.4,
dated 2022-05-09, at
[`252644c0d347a663ea5d7bef88fa2ab04f114b1e`](https://github.com/cran/intccr/tree/252644c0d347a663ea5d7bef88fa2ab04f114b1e).
Its DESCRIPTION declares GPL >=2. Downloaded source remains ignored under
`research/raw/intccr/`; the following Git blob hashes were verified locally.

| File | Git blob |
| --- | --- |
| R/ciregic.R | `e1f2f6e29d84e6f520e0482adfdcc0a48c8f9b90` |
| R/bssmle.R | `d122c3ac22a446bfb8071b720c9df962de188ebe` |
| R/Surv2.R | `c32f52d7c8c275cda51dd23e9868d9d7e2615fd9` |
| R/bsderivs.R | `e468f3fd071e490b6e57f94ffc490776ab56752a` |
| R/naive_b.R | `bb46420a92a19bd634f2db6d69e5edb0d6677248` |
| R/bssmle_lse.R | `416b818d2e826169e20f917990193736011fa012` |
| R/dataprep.R | `371ac55900e8434a264c557e1933b749f3918576` |

## Required statistical behavior

- Outcomes are right censoring (0), cause 1 or cause 2. Event intervals have
  strictly ordered endpoints. Exact events are not part of this source API.
- Each cause has its own numeric regression coefficients, cubic B-spline
  baseline and nonnegative generalized odds-rate parameter `alpha`. Zero gives
  the proportional subdistribution hazards link; one gives proportional odds.
- For `eta = B(t) phi + x beta`, the cause-specific CIF is
  `1 - (1 + alpha exp(eta))^(-1/alpha)`, with the continuous `alpha=0` limit
  `1 - exp(-exp(eta))`. Stable differences and log probabilities are necessary.
- An observed cause contributes its CIF increment over `(v,u]`; left-censored
  events with `v=0` contribute the CIF at `u`. Right censoring contributes
  `1-F1(v)-F2(v)`, and its supplied upper endpoint is irrelevant to this
  likelihood and to knot placement.
- Baseline spline coefficients are ordered within each cause. Joint
  probability constraints must enforce `F1+F2 <= 1` over the supported covariate
  domain. The source constrains the upper time boundary at all corners of the
  observed covariate range and imposes zero incidence at its lower boundary.
- Knot count is `floor(k * length(c(v,u[event>0]))^(1/3))`, `0.5 <= k <= 1`.
  Fitting takes unique interior empirical quantiles; boundaries are the minimum
  and maximum of that same endpoint vector. The initializer uses different
  quantiles, so duplicated knots can make its dimension differ from the fit.
- `nboot=0` still computes a regression covariance: individual regression scores
  are projected off the baseline score span, and the residual score crossproduct
  is inverted. This is not the joint likelihood Hessian. Rank checks and stable
  least-squares solves are needed before claiming equivalent uncertainty.
- Native predictions reject times outside the observed boundary range. The
  author contour requests 50 times from zero to the upper boundary, which needs
  an explicit decision when the fitted lower boundary is positive.

## Source concerns to resolve with numerical evidence

The native inequality Jacobian multiplies both cause blocks by the sum of the
two link derivatives. Differentiating the stated constraint gives one cause's
own derivative in its respective block. The equality Jacobian also places zeros
in the baseline coefficient columns despite differentiating a baseline-boundary
constraint. These are source-level discrepancies; finite-difference checks and
an executed optimizer reference are still needed before assessing their effect
on fitted results.

`Surv2` checks `v >= u` before applying the documented rule that a censored
observation may have an arbitrary or missing upper endpoint. The Python input
contract should follow the likelihood's censoring semantics explicitly.
The long-format helper also sorts by `ID & time`, not lexicographically by ID
and time; any future converter must establish visit order before selecting an
event interval. Neither behavior should silently become a Python convention.

For a source-only reference run, the pure-R optimizer dependencies have also
been retrieved and blob-verified without installation: `alabama` 2025.1.0 at
`3dd535fac47afe823162a4755c3a1faddef6c566` (`R/constrOptim.nl.R`, blob
`05c93764ceb3eacafd1d2cc06ad8d6b4f7015cf1`), and `numDeriv` 2016.8-1.1 at
`54dc4181ec0543a95a2cf7a5e3c483ab0a109750` (`R/numDeriv.R`, blob
`394d5f1db1d644fd1b44a2e70f0294ce04fcf50d`; `R/num2Deriv.R`, blob
`9adbafd4d39dc95d2d0b69283a5fc571bf1e497c`). The reference harness redirects only
the optimizer's package lookup to these source-loaded functions. No extra R
or Python dependencies have been installed.

## Executed native fits and derivative checks

`tools/reference_interval_competing_risk.R` sources the pinned routines,
without editing the likelihood or fitting-body mathematics. Three deterministic
120-row datasets share inputs and vary `alpha` over `(0,0)`, `(1,1)` and `(0,1)`.
All three runs report native convergence. Their likelihoods are respectively
`-277.68320009617702`, `-277.02031996922011` and `-278.03084011105767`.
The preserved fixtures contain 120 input rows, 48 parameters, 48 covariance
entries and 279 pairs of predicted cause-specific CIFs.

Independent numerical differentiation at an interior parameter point confirms
the constraint derivative discrepancies: maximum inequality-Jacobian errors
are `0.207`–`0.241` and equality-Jacobian errors `0.01165`–`0.01168`. The
likelihood gradient agrees within `5.55e-7`. The native fitted lower-boundary
equality residual is `3.78e-7`–`4.72e-7`; its first baseline controls are around
`-15` and `-22`. Thus native fitting uses a finite approximation to the
mathematical zero-incidence boundary, which requires an infinite log parameter.

A second run keeps the same model and native optimizer but supplies independent
analytic constraint Jacobians, verified against finite differences within
`1e-7` at the evaluation point. The likelihoods become `-294.90545106562894`,
`-278.20886548029046` and `-282.560590217423`, with lower-boundary residuals
`3.60e-11`, `3.55e-8` and `7.94e-9`. These materially different results are
diagnostics, not certified optima or Python acceptance targets. Ordinary
objective scores alone are not KKT residuals in a constrained problem.
Diagnostics for both runs are preserved separately in the JSON fixture.

The original reference run took 3.59 seconds and peaked at 99.4 MiB child RSS.
The combined original/corrected run took 28.82 seconds and peaked at 112.0 MiB.
Both reported zero process swaps and used one numerical thread. A preliminary
fully finite-difference optimizer comparison was explicitly terminated at its
45-second experiment limit; it produced no corrected fit and is not evidence
of a successful fit.

The planned Python interface exposes `boundary_cif_tolerance`, defaulting to
`1e-7`. Inverting the link makes each lower-boundary constraint linear in the
spline and regression coefficients; this avoids optimizing near-zero raw
equality residuals and makes the finite approximation explicit. Monotone CIFs
need joint probability constraints only at the maximum supported time. Source
corner constraints do not by themselves certify every interior covariate
profile for arbitrary link parameters; prediction must check the requested
profile's joint probability and report violations. No Python fit or numerical
equivalence claim has yet been made.
