# Parametric survival regression and contours

The previous goal turn integrated right-censored Fine–Gray regression and
contours at `d97bf9d`, with numerical, package and plot checks. That turn made
implementation progress. This checkpoint targets additional statistical
methods in partial SurvivalContour entry 166, rather than adding CI machinery.
Catalog totals are 62 implemented, 66 partial and 10 pending.

One Luna worker implements in the separate existing checkout on
`feat/parametric-survival-luna`; root coordinates and validates on
`feat/parametric-survival`. Numerical jobs are serial and use one BLAS/OpenMP
thread. No installation, full-suite run or publication retry is planned.

## Verified method scope

The [official app](https://biostatistics.mdanderson.org/shinyapps/survivalContour/)
was read again on 2026-09-27. Version 1.0.2 lists Weibull, log-normal,
log-logistic, original generalized-gamma and stable generalized-gamma AFT
models, along with a spline model. Covariates act on the location parameter.
The current implementation checkpoint covers the first three exact location-
scale families. Both generalized-gamma parameterizations and splines remain
required future work. Existing ACCFLF bounded limiting degrees of freedom and
conditional shape covariance must not be relabeled as these exact models.

The author SurvivalContour source is pinned at
[`d4645f69f23fc1146c07432f576b4c40f85e1bba`](https://github.com/YushuShi/survivalContour/tree/d4645f69f23fc1146c07432f576b4c40f85e1bba).
`R/paraContour.R` has Git blob `39e141a400d440097946571bf15eba3c6061b6cc`,
and `R/paraContour3D.R` has blob `06ca05a05a45063f287399d782a513ae05ad2b87`.
Both use a continuous-covariate grid between the empirical 2.5th and 97.5th
percentiles (30 points by default), mean or explicit adjustment profiles,
and distinct event times with zero prepended. Both call `predict.flexsurvreg`;
the 3D helper passes point estimates and lower/upper bounds to its renderer.

The numerical parameter and prediction source is `flexsurv` 2.3.2, pinned at
[`2aae4c8ac56823d0eac30c1a9ad654ac599b5938`](https://github.com/cran/flexsurv/tree/2aae4c8ac56823d0eac30c1a9ad654ac599b5938).
Exact source bytes are saved only under ignored `research/raw/flexsurv`, with
their Git blob hashes verified:

| File | Git blob |
| --- | --- |
| DESCRIPTION | `22a97588ac5f02a819cf6eeae42a4e055260f5f1` |
| R/distributions.R | `ec1fcfd33cdc35c6dbc00aba702208267c8c7cda` |
| R/flexsurvreg.R | `14cb269b8d1213cdfb96d7dc4e63f5d6d033e532` |
| R/predict.flexsurvreg.R | `df27ce40dc9ce485416f1e87daf61c074874c2f9` |
| R/summary.flexsurvreg.R | `3c91ccb838e176ee5218e81045d74cfb5c50d430` |
| R/GenGamma.R | `80f0586694cccd42ae9db07043a15445e672853c` |

`R/distributions.R` maps all three current families to `survival::survreg`
initial estimates. The Weibull and log-logistic distribution shape is
1/sigma, with distribution scale exp(mu); log-normal has mean mu and
standard deviation sigma on the log-time scale. Here sigma denotes the positive residual scale
in `log(T) = mu + sigma Z`, not log(sigma). Python's joint covariance coordinates
are intercept, numeric slopes, then log(sigma), matching `survreg`'s covariance.

Native flexsurv curve bounds simulate from the asymptotic joint normal
parameter distribution and take survival-probability quantiles (default 1,000
draws). The planned deterministic Python bounds use the delta method on log
cumulative hazard. They are a distinct interval method, not native Monte-Carlo
interval parity. Fitting and point prediction comparisons are unaffected.

## Native references

`tools/reference_parametric_survival.R` executes installed `survival` 3.6-4
without additional packages. Forty-eight deterministic observations include
numeric covariates, tied times and mixed exact/right-censored observations.
Six native fits cover all three families with two covariates and with only an
intercept. Three CSV fixtures retain 144 scalar fit metrics and 2,160 prediction
rows: parameters, full joint covariance, observed information, time-density
log likelihood, survival, log survival, cumulative hazard and pointwise bounds.

Prediction rows include five profiles, mean/explicit adjustment values and
native/custom time grids. Time zero, early survival near one and late tails are
included. Reference delta-method gradients use a five-point numerical
derivative of R's log cumulative hazard and the native fitted covariance.
The generator ran successfully; it does not claim to execute the full flexsurv
or SurvivalContour formula/UI stack. Python implementation verification follows
after integration.

## Core integration and independent validation

Luna implemented the exact three-family AFT core in `b2895c3`, integrated as
`922090f`. Root review corrected draft Hessian signs, the normalized log-time
Jacobian in the reported likelihood, direct transformation of observed
information, the log-logistic/log-normal log-hazard calculations and unnecessary
profile-by-time-by-parameter allocations. The final prediction contracts the
full coefficient/scale covariance using profile-by-time arrays and preflights
a combined array budget. Input units are normalized before design-rank checks.

Root independently reproduced an unbounded location estimate: three exact
events in one binary-covariate level and three censors in the other appeared to
converge with increasingly large group effects. A bounded LP on the normalized
design now rejects directions that leave every event location unchanged and
raise some censored locations without lowering any. This check is independent
of an optimizer's small-gradient stopping rule. A focused regression confirms
rejection for all three distributions.

Two focused tests passed in 1.42 seconds. The root adapter independently checked
all six native fits, 144 scalar metrics and 2,160 prediction rows. Additional
checks covered covariate units 1e-100/1e100, common covariate offsets of 1e8,
time units 1e100, the exactly zero contribution of a censor at time zero, and
prediction times 1e-200/1e200. Maximum absolute errors were 5.78e-10 for
coefficients/log-scale, 2.08e-11 for covariance, 1.66e-7 for information and
5.69e-14 for log likelihood. Survival and pointwise-bound errors were below
8.65e-10; log-cumulative-hazard standard-error differences were below 8.01e-9.
The maximum absolute log-survival/cumulative-hazard difference was 0.00427 at
a cumulative hazard above 618,000, within the checked relative tolerance.

The independent numerical audit took 0.057 seconds after imports, peaked at
115.5 MiB RSS and reported zero process swaps. These measurements describe the
small reference workloads. No full suite, new CI configuration or installation
was used.

## Contour and public API integration

Luna's contour wrapper (`d0af6ab`, integrated as `7f3224b`) fits once and returns
both the continuous grid and selected covariate-percentile curves. Root exposed
the six public interfaces and extended the existing plot type annotations to
accept these results. The plotting implementation is shared with Cox contours.

An independent root adapter compared twelve surfaces against native references:
three distributions, mean/explicit profiles, and native/custom time grids.
Probability and bound errors remained below 8.65e-10. The adapter verified
default percentile profiles and their predictions, executed both guide examples,
and rendered and visually inspected the 2D survival contour, 3D survival surface
and 3D lower-limit surface. Peak RSS was 154.8 MiB with zero reported process
swaps. The six focused parametric/shared-contour tests passed in 1.45 seconds;
the added memory regression verifies oversized output rejection before fitting.
Scoped Ruff and mypy checks passed after formatting the contour wrapper.

## Package and completion checkpoint

The cached Hatch backend built wheel and source archives. All 469 Python source
modules matched both archives byte for byte; the wheel catalog, attribution
notices and license files matched the workspace. The source archive includes
the guide, audit, reference generator, three CSV fixtures and focused tests;
ignored native-source downloads and scratch output are excluded. An isolated
interpreter loaded the wheel, checked all six new public exports and executed
both guide examples. That run peaked at 137.7 MiB with zero process swaps.

These results complete this three-family checkpoint, not the full catalog.
Entry 166 remains partial; totals remain 62 implemented, 66 partial and 10
pending. Both generalized-gamma parameterizations, splines and other recorded
method gaps remain required. The preceding user-status response was a status
restatement, not implementation progress; this continuation completed the
public integration and verification. The overall goal remains active.

Automatic approval review previously rejected GitHub publication because it
requires approval unavailable in this session. No publication retry or bypass
was attempted. Local method implementation can continue independently.
