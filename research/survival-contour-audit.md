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

## Integrated ordinary Cox workflow

Luna commit `395ca37` was integrated as `11f5417`. Public entry points are
`survival_cox_contour`, `SurvivalCoxContour`, `plot_survival_contour_2d` and
`plot_survival_contour_3d`. Results retain fit/tie metadata, confidence level,
adjustment profile, grid, times, five primary prediction arrays and five
covariate-quantile prediction arrays. Plot imports are lazy. The old
`survan_cox` Breslow default remains unchanged; Efron is an explicit new option.

Root review identified and the implementation addressed separate event/censor
weight scaling, log-space baseline accumulation, standard errors without
squared-hazard intermediates, and rejection of excessive output before the
fit starts. Root then moved the existing two-million-design-entry constraint
before normalization as well. Inputs are bounded to 100,000 observations and
100 covariates; a combined two-million-cell surface/profile/vector budget is
checked before fitting. No dependencies or CI configuration were added.

Six targeted existing/new Cox checks passed in the worker (1.25 seconds),
including Efron score/information finite differences with multiple tied deaths,
extreme tied event/censor predictors and existing original-SURVAN regressions.
Worker Ruff format/check, targeted mypy and diff checks passed. The worker also
compared all four original-R surface and quantile-curve scenarios. Root's
independent integration check, with warnings as errors, compared all four fits,
coefficient covariances, likelihoods, primary surfaces and quantile curves.
Maximum probability/limit error was `7.78e-16`; maximum cumulative-hazard or
log-survival-standard-error error was `5.11e-15`.

Additional root checks covered simultaneous time scaling by `1e100` and
covariate scaling by `[1e100, 1e-100]`, right-continuous values between events,
censor-only plateaus, survival one before the first event, flat survival after
the final event, and read-only results. The integrated numerical check took
0.055 seconds after imports and peaked at 115.3125 MiB with zero reported swaps.

A direct baseline/prediction check used predictors at +1000 and -1000 with a
low-risk late event: both Efron and Breslow produced the analytic survival
`exp(-1)`, hazard 1 and standard error 1 at the final time, despite individually
unrepresentable baseline hazard and variance in ordinary units. Oversized
output was rejected before a patched fit could run; an oversized design was
likewise rejected before fitting. The 2D contour and 3D surface were rendered
and visually inspected together. That process peaked at 161.453125 MiB and
reported zero swaps. Plot interpolation is documented separately from the
right-continuous numerical predictions.

The root's final preflight edit passed its targeted check and Ruff after line
formatting. No full suite, large simulation or package installation was run.
Catalog entry 166 advances to **partial**, with all other model families still
listed explicitly. Totals are now 62 implemented, 66 partial and 10 pending.
