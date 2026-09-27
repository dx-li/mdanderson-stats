# SurvivalContour implementation audit

Baseline main `a80d81c` is clean and contains the completed BOIN/EasyCellType
checkpoint. The preceding goal turn is progress: new methods, original-R
references, packaging checks and coverage metadata were committed. The whole
catalog/publication goal remains active. Current totals are 62 implemented,
65 partial and 11 pending.

Root works on `feat/survival-contour`, with one Luna worker on
`feat/survival-contour-luna` in the existing independent checkout. Numerical
jobs are serial and use one BLAS/OpenMP thread. No installation or full-suite
run is planned. The [source scouting record](survival-contour-next-audit.md)
identifies the pinned author repository and the full app scope.

## Initial implementation contract

The first family is ordinary right-censored Cox analysis, prediction surfaces
and optional 2D/3D plotting. The existing `survan_cox` fit will gain optional
Efron ties while retaining its legacy Breslow default. The new contour workflow
will default to Efron, matching an ordinary R `coxph` call. This reuses the
existing scaling, optimization, information and separation machinery rather
than creating an unrelated second fitter.

The author helper uses 30 grid points evenly spaced between the continuous
covariate's empirical 2.5th and 97.5th percentiles. R's default type-7 quantile
corresponds to linear interpolation. Other numeric columns are set to means,
or an explicit profile can be supplied. Python will require an already numeric
design matrix; categorical encoding is the caller's responsibility. Predictions
are covariate-by-time matrices on all distinct observed times, including
censor-only times. An optional time grid will use right-continuous steps.

The app advertises five covariate-quantile curves, but its exact quintet is not
specified in the available author helper. The Python summary probabilities
(.1, .25, .5, .75, .9) are explicit and configurable. These are quantiles of
the predictor, not quantiles of survival time.

## Prediction and interval formulas

Installed R `survival` 3.6-4 is available without installation. Its
`agsurv`, `coxsurv.fit`, `survfit.coxph` and `survfit_confint` bodies were
inspected directly. The author's
[agsurv5.c](https://github.com/therneau/survival/blob/master/src/agsurv5.c)
was also read, with Git blob `f7cc11c5dcc5b15a530445f6f7e39ad23d8b644d`.
The executable fixtures pin the installed package, rather than assuming that
every current development source matches it.

At an event time with d deaths, write S0/S1 for risk-weight/weighted-covariate
sums and D0/D1 for the death sums. For each j from 0 through d-1, q=j/d for
Efron and q=0 for Breslow. With denominator S0-qD0, accumulate baseline hazard
by its reciprocal, independent variance by its reciprocal square, and the
baseline coefficient-gradient vector by (S1-qD1)/denominator squared.
For profile z and relative hazard r, H=r H0 and

```text
Var(H) = r^2 [V0 + (H0 z - A)' Cov(beta) (H0 z - A)].
```

Consistent centering/scaling leaves these predictions unchanged. Survival is
exp(-H); the default pointwise log confidence bounds are
exp(-H +/- zcrit sqrt(Var(H))), with the upper bound capped at one. These are
pointwise limits and do not establish simultaneous confidence for a surface.

## Executable reference checkpoint

`tools/reference_survival_contour.R` loads the unchanged, checksum-verified
author `coxContour.R` and `cox3DContour.R`. Only plotting sinks are replaced,
to capture their predictions without installing plotly or the larger package.
Four cases cross Efron/Breslow with mean/explicit profiles, each on five grid
points and nine observed times. Fixtures also record coefficient covariance,
standard errors, p-values, likelihoods, cumulative hazards and log-survival
errors. Direct R `survfit` calls provide the additional five quantile curves.

The generator completed in 0.64 seconds. Its first attempt stopped because R
warned that the requested optimizer convergence tolerance was below the default
Cholesky tolerance; the generator then set both explicitly (`eps=1e-12`,
`toler.chol=1e-14`) and completed with warnings as errors. No model or dataset
was changed to obtain success. Four CSV fixtures preserve input, fit, surface
and quantile-curve references.

Stratified and interval-censored Cox, parametric/spline models, Fine–Gray,
forests, neural models and full native app workflows remain later coverage
requirements. Entry 166 is still pending until this implementation is integrated
and verified. Final API and validation results will be recorded below.
