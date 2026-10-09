# SYNERGY original parametric model recovery

On October 8, 2026, the direct institutional `SYNERGY_V3.zip` download
succeeded. The 26,499-byte archive has SHA-256
`23ce39811739b1f856910d0974403815bdaf423eb994db34f0496960460daa51`.
It supplies five S-PLUS/R listings, the tutorial example, published input
table and Word readme. Authors are J. Jack Lee, Maiying Kong, G. Dan Ayers
and Reuben Lotan. Source/member pins are in
`docs/synergy-parametric-source.json`.

This recovery supersedes the earlier archive-access failure in
`synergy-response-surface-audit.md`. The four parametric surfaces are now
implemented and validated; the separate 2008 semiparametric bootstrap
interval convention remains unresolved, keeping catalog entry 18 partial.

## Equations and workflow

The original wrappers transform fraction surviving to its logit and fit
unweighted nonlinear least squares. Greco, Machado and Plummer omit controls
with both doses zero. Carter instead omits boundary responses and includes
interior controls. Initial values are obtained by separate single-drug
regressions; Carter uses raw doses and the other three use log doses.
Native interaction starts are alpha=0, eta=0.7, beta4=2.5 and beta12=1.

The Greco interaction term is proportional to the **product** of normalized
doses, with the odds-ratio exponent given by half the sum of reciprocal
slopes. It must not be replaced by a geometric mean of normalized doses.
For equal slopes m, its closed form is
`logit(E)=m*log(A+B+alpha*A*B)`. An independent closed-form test verifies this
amplitude as well as the equal-slope Machado and constant-potency Plummer
limits. All four models' interaction direction is checked on the fraction-
surviving scale, including negative Greco interaction.

Python provides response evaluation, fitting, complete local parameter
uncertainty, residual diagnostics, control-aware prediction and exportable
single-drug/fixed-ratio/contour figures. Original scripts' equations are
implemented independently. External original code, documents and binaries
remain ignored research inputs and are excluded from distributions. No express
source license was found in the archive readme/listings; no upstream source
redistribution grant is inferred from their public availability.

## Independent original-program references

`tools/reference_synergy_parametric.R` verifies the four original listing
checksums, sources them unchanged and runs base R 4.5.0. Only plotting callbacks
are replaced with no-ops, avoiding GUI/device output during reference generation;
mathematical functions, initialization and original nls wrappers are unchanged.

The committed CSV references contain:

- 138 original-kernel dose predictions, including unequal/equal slopes,
  marginal axes, synergy and valid negative Greco interaction;
- nine original-wrapper fits: all four models on the published data and on
  synthetic data, plus an antagonistic Greco example;
- 43 parameter estimates, starts, standard errors, t statistics and p-values;
- 207 covariance/correlation values, with residual error/SSE/df records.

Maximum absolute kernel response discrepancy is `1.249e-15`. Among the first
eight fits, maximum absolute parameter difference was `8.73e-5` and maximum
relative standard-error difference was `1.72e-5`. Original R nls uses its
default `1e-5` convergence tolerance; Python refines further. Tests compare
estimates with `rtol=3e-5, atol=2e-5`, standard errors with
`rtol=3e-5, atol=1e-8`, covariance with `rtol=1e-4, atol=1e-7`, and residual
SSE/RSE with `rtol=1e-9`. The antagonistic fit also passes these tolerances.

Thirty-two method tests and four figure/export tests pass with warnings
treated as errors. Together with existing semiparametric and WFMM prior tests,
all 80 focused tests pass. PNG figures were generated for every model and
validation errors leave no extra open figures.

## Numerical and compatibility limits

The native Greco/Machado kernels bisect E a fixed 50 times; Plummer applies
30 unverified Newton steps. Python solves the implicit equations in log
coordinates with bounded bisection and verifies the residual equation.
Decreasing marginal response is required for these three biological models.
Supported parameter/data/work bounds are listed in the public guide.

For Greco, let `c1=-1/m1`, `c2=-1/m2`, and positive normalized doses A,B.
A sufficient derivative condition for a unique monotone implicit response is
`alpha*sqrt(A*B) > -4*sqrt(c1*c2)/(c1+c2)`.
The negative-interaction constraint depends on the dose grid. Prediction
rejects grids violating it; fitting parameterizes alpha above its grid-specific
limit and converts the optimizer Jacobian back to physical parameters before
forming covariance. This supports the verified antagonistic fit without
silently selecting an ambiguous branch at much larger new doses.

For Plummer, beta3>-1 ensures the log-scale potency equation is monotone;
beta4>-2 keeps the effective combination dose positive. Native unconstrained
nls can leave these domains. Python rejects boundary/failed fits, zero
residual variance and singular or ill-conditioned parameter information.
Carter is linear on the logit scale and is solved directly.

Plot origins in the three log-dose models use the observed control mean when
available, matching the source callbacks. The direct response API's theoretical
origin remains one; this is distinct from including controls in the fit.
Native plot artwork, R summary object schemas and exact optimizer trajectories
are compatibility differences. They do not create an additional recovered
parametric calculation.
