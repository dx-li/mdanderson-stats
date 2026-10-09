# SYNERGY wild-bootstrap workflow audit

## Source contract implemented

Kong and Lee (2008), “A Semiparametric Response Surface Model for Assessing
Drug Interaction,” *Biometrics* 64:396–405, DOI
10.1111/j.1541-0420.2007.00882.x, describes the two-stage procedure recorded in
`synergy-response-surface-audit.md`: estimate the source smoothing parameter
lambda, construct residuals with the half-lambda spline, center pseudo-data on
the twice-lambda fit, draw independent Mammen weights, and refit baseline plus
REML spline for each replicate. The code implements the described transformed-
scale formulas on all training rows, with independent weights by observation
row, including duplicate-dose and marginal rows.

The public API is `bootstrap_synergy_surface`, returning
`SynergySurfaceBootstrap`. It returns original departure, optional draw matrix,
row-wise draw mean and ordinary sample SD, plus per-replicate lambda, REML
objective, scaled residual variance/response scale, optimizer evaluation count
and boundary flag. It neither forms confidence limits nor claims native
interval equivalence. Draw SD uses the explicit Python descriptive convention:
center on the sample mean with denominator `B-1`. This is deliberately labeled
as a convention because the source's exact interval SD centering and denominator
remain unverified.

The API accepts a seed/generator or a bounded row-by-replicate Mammen multiplier
tape for deterministic replay, performs refits serially, rejects failed or
nonconverged refits with their index, and rejects a log-dose dataset containing
a both-zero row before randomness is consumed. It enforces a conservative
aggregate fitting-work limit, a 2,000-replicate limit, and one-million-cell caps
for retained draws and tapes. Automatic fits use the same lambda semantics and
penalty units as `fit_synergy_surface`; half/double anchors therefore use
lambda/2 and 2*lambda without response-scale adjustment.

## Validation

`tools/reference_synergy_surface_bootstrap.R` independently constructs the raw-
dose baseline and fixed-lambda spline with the augmented penalized system, and
profiles each original/replicate REML fit through full covariance matrices.
It creates five fixed row-wise Mammen tapes across 13 observations, including
repeated dose pairs, and writes the four CSV fixtures in `tests/fixtures/`.
The measured maximum relative differences were `1.64e-7` for the original
lambda, `3.37e-7` for replicate lambdas, `3.37e-7` for departures, and
`8.59e-8` for sample SD; the maximum absolute departure difference was
`1.30e-10`. Tests allow `3e-7` for the original lambda, `5e-7` for replicate
lambdas and departures, and `1e-7` for SD, with a small absolute floor near
zero. The Python test compares original/half/double fits, replicate lambdas,
all departures, draw mean and `ddof=1` SD against the independent files.
Additional focused tests check summary-only storage, log-
dose refits, invalid tape rejection, both-zero rejection before RNG use and
work-budget rejection before RNG use. No R confidence interval is added because
the source SD convention is unresolved.

## Remaining limits

The separately audited parametric surfaces are now covered. The source interval convention, native
random stream, case-study reproduction, and native reports/plots remain
unimplemented. This addition alone does not complete SYNERGY catalog entry 18.

## Integrated checkpoint

All ten focused surface/bootstrap tests passed with warnings treated as errors.
The same five-tape workflow was also checked after multiplying responses by
`1e-200` and `1e200`. After undoing the unit change, the maximum absolute
differences were `1.06e-10` for the mean and `6.53e-12` for the sample SD.
The combined check took 2.106 seconds, peaked at 149.36 MiB and reported zero
process swaps. Targeted Ruff checks and formatting passed; the implementation
checkout's targeted mypy check also passed. The full local suite was not run;
the existing GitHub workflow supplies the broader checks after publication.
