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
1/sigma, with distribution scale exp(mu); log-normal has log-mean mu and
log-standard-deviation sigma. Here sigma denotes the positive residual scale
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
