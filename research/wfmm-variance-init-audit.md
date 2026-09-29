# WFMM variance initialization audit

The recovered WFMM implementation accepts explicit starting values for each
coefficient's random-effect and residual variances. Its empirical-Bayes helper
calibrates fixed-effect shrinkage conditional on those variances; it does not
estimate them. The source audit did not recover the native automatic
initialization algorithm or the mapping from `delta_omega` to the
inverse-gamma prior parameters. That missing native formula and its scale and
boundary conventions prevent a defensible claim of native parity.

`initialize_wfmm_variances` therefore exposes a separately labelled Python
policy. For each coefficient it maximizes the restricted Gaussian likelihood
under the covariance already implemented by WFMM,

`V = sum_g(q_g Z_g Z_g.T) + diag(s[residual_strata])`.

It rejects nonpositive residual degrees of freedom, deficient fixed designs,
and covariance components that cannot be distinguished after projection away
from the fixed-effect space. A single residual component has the closed-form
REML estimate; multiple components use bounded, serial, per-coefficient
optimization with an explicit likelihood-evaluation limit. Reported starts
include raw estimates, optimizer status, boundary indicators, and residual
diagnostics. A scale-relative positive floor is applied only to sampler starts
and never changes the reported raw estimate or defines a prior. All-zero data
have no internal variance scale and require a caller-supplied absolute floor
for usable starts.

Independent checks use the closed-form residual-only estimator and balanced
random-intercept ANOVA REML equations, with likelihoods recomputed from the
resulting covariance in base R. These validate the implemented model and
Python policy, not the undocumented native initializer.


## Parameterization and reported likelihood

The reported restricted log likelihood is
`-0.5 * [(N-P)*log(2*pi) + log|V| + log|X.T V^-1 X| + r.T V^-1 r]`,
where `r=y-X*beta_GLS`. Fixed-column normalization is undone in the
information determinant as well as the coefficient estimates. Coefficient
response scaling contributes `(N-P)*log(data_scale**2)` to the deviance.
`data_scale` consistently records `max(abs(y))`, not a variance.

For mixed or stratified models, each variance is parameterized as
`exp(log_ratio) * data_scale**2 / (C * mean(diag(B_component)))`, with
`C` covariance components and fixed `log_ratio` bounds `log(1e-8)` to
`log(1e8)`. Returned raw values retain finite optimization bounds; bound flags
are not a claim that a zero or unbounded-parameter optimum has been found.
The best valid likelihood evaluation is retained if an explicit budget is
exhausted, and optimizer success is false. Positive starts alone do not imply
convergence. The default limits are 500 iterations and 2,000 actual likelihood
evaluations per coefficient; the evaluation guard also covers numerical
finite-difference calls. The global preflight bounds are 250 million estimated
initialization work units, 50 million covariance-rank work units and eight
million projected covariance-basis cells. The analytic residual-only path
charges its least-squares work and avoids a dense projected covariance check.

The numerical-zero threshold is `8*eps*max(N,P)*max(1,norm(y/data_scale))`.
Such cases retain the residual norm, return raw zero variance with a numerical
boundary status, and report no finite restricted likelihood (`NaN`). All-zero
data similarly report `NaN`; an explicit absolute floor supplies optional
starts without inferring a statistical variance scale.

## Integrated validation

Five focused tests match independent base-R residual-only and balanced
random-intercept REML results, including reported likelihoods. They also cover
rank-deficient and confounded designs, all-zero/numerical-zero outcomes,
finite-design rescaling and actual likelihood-budget exhaustion. Root tightened
the tiny coefficient comparison to zero absolute tolerance so a returned zero
cannot pass a nonzero `2.5e-200` reference.

All six guide blocks pass. A separate two-coefficient balanced model estimates
random variances `[1.6266715171, 0.4066666645]` against analytical
`[1.6266666667, 0.4066666667]`, and residual variances
`[0.08000024452, 0.01999999780]` against `[0.08, 0.02]`, within `1e-5`
relative tolerance. Its two optimizations use 30 and 36 likelihood evaluations.
Those returned starts feed the existing fixed-variance sampler successfully.
Rescaling the fixed design by `1e-200` preserves fitted coefficients after
unit conversion and shifts restricted log likelihood by the specified
Jacobian constant. Combined root checks take 2.755 seconds at 148.72 MiB peak
RSS with zero swaps, using one process and single-thread numerical libraries.
The native initialization/prior mapping remains open; entry 70 stays partial.
