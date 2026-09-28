# Spline-survival Monte Carlo uncertainty

The source is flexsurv 2.3.2 at commit
`2aae4c8ac56823d0eac30c1a9ad654ac599b5938`, shared with the
[spline fit audit](survival-spline-audit.md) and the
[parametric Monte Carlo audit](survival-uncertainty-audit.md).
`cisumm.flexsurvreg` obtains joint-normal parameter draws through `normboot`
and evaluates the requested summary for each draw. It uses pointwise R
type-7 quantiles with missing values removed; ordinary sample SD propagates
missing values. It does not redraw or filter spline coefficients by slope.

For positive finite time, `psurvspline` applies the requested survival link
to the spline predictor: `exp(-exp(eta))`, `plogis(-eta)`, or `pnorm(-eta)`.
Its source forces survival to one at time zero and zero at positive infinity.
These endpoint overrides apply even when a sampled coefficient vector has
negative tail slope. The Python calculation should preserve the small
survival tail directly rather than the source wrapper's CDF round trip.

## Unrestricted draws are not necessarily proper survival models

A full Gaussian coefficient distribution can produce negative spline slopes,
and therefore a rising survival curve on part of the time axis. The source's
prediction helper does not check slopes. Its fitting density only zeroes
invalid derivatives at the evaluated event times, while the Python fitter
checks global positivity for the fitted model.

Retaining finite nonmonotone simulated curves reproduces the source's
unrestricted Monte Carlo summary. This is not a claim that each draw defines
a proper survival distribution. Filtering or conditioning draws on monotonicity
would change those interval estimates. The Python extension exposes the
minimum spline slope for each draw in normalized log-time coordinates, while
keeping `valid_draws` as a count of nonmissing numerical evaluations. Endpoint
overrides and the scope of these diagnostics must be stated explicitly.

## Independent reference fixture

`tools/reference_survival_spline_uncertainty.R` reads the six previously
verified native-kernel fit/covariance references: hazard, odds and normal links,
each with zero or four internal knots. It generates 64 joint-normal draws in
original parameter coordinates for each fit and replaces one draw with a
deliberately negative linear slope. This ensures that the reference cannot
pass if a port silently filters rising curves. The first draw retains the
reference fitted parameters.

An independent base-R truncated-power spline basis evaluates three covariate
profiles at ten times, including both endpoints. R logistic/normal probability
functions and type-7 quantiles supply pointwise limits and sample standard
deviations. The fixture contains 2,304 parameter cells and 180 summary rows.
Generation completed successfully in .50 seconds without a package installation
or compilation.

The integrated Python predictor matches all 180 rows, with maximum absolute
lower-limit, upper-limit or sample-SD error `2.759e-14`. All 64 draws remain
evaluable at each reference point, including the deliberately rising curve;
its reported minimum slope is negative. An independent affine transformation
of time and covariates, the complete joint covariance and the same supplied
draws preserved fitted curves, limits and SDs within `1.953e-14`. Minimum
slopes changed by the expected log-time scale factor. This audit took .091
seconds after import, peaked at 116.7 MiB RSS and reported zero process swaps.

The basis is cached for positive finite-time cells and reused across profiles
and draws. Exact zero/infinite endpoint overrides never evaluate a spline
basis, avoiding unnecessary invalid arithmetic for tiny knot spacings. Work
bounds include both prediction basis width and per-draw global slope checks.
The shared contour wrapper still uses deterministic intervals at this
checkpoint; its Monte Carlo integration remains separate work.

The public 100-row, four-knot example exposed small asymmetric roundoff in
the fitter's inverse Hessian, which caused generated-draw covariance validation
to reject an otherwise positive-definite fit. The fitter now symmetrizes its
scaled and original-coordinate covariances and checks positive definiteness.
The spline/uncertainty tests pass (nine), as do focused lint and type checks.
The guide's fit and 500-draw prediction now pass together in 1.178 seconds,
using 116.1 MiB peak RSS and zero process swaps. Three draws in that seeded
example have negative minimum slope; the diagnostic reports them explicitly.
