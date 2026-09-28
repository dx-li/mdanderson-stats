# Complete-data log-normal Bayesian diagnostic

The cached BCSTTE user guide §4.5 parameterizes a log-normal variable by
location `mu` and standard deviation `sigma` of `log(X)`, with
`log(X) ~ Normal(mu, sigma²)`. The guide describes the distribution and a
family-level goodness-of-fit workflow, but does not specify the executable's
log-normal prior or fitting algorithm. This implementation therefore uses an
explicit, proper Normal-Inverse-Gamma prior and makes no claim of native prior
or fitter parity.

For `Y=log(T)`, use `sigma² ~ InvGamma(a0,b0)` (shape/scale) and
`mu|sigma² ~ Normal(m0,sigma²/kappa0)`. With `n` complete observations,
`ybar=mean(Y)` and `SSE=sum((Y-ybar)²)`, the conjugate posterior parameters are
`kappa_n=kappa0+n`, `m_n=(kappa0*m0+n*ybar)/kappa_n`,
`a_n=a0+n/2`, and
`b_n=b0+SSE/2 + kappa0*n*(ybar-m0)²/(2*kappa_n)`. The posterior draws are
joint: `sigma²` is inverse-gamma and `mu` is conditionally normal. Each observed
time is evaluated with that same paired parameter draw before the existing
Johnson CDF diagnostic is applied. The result retains `centered_location_samples`
with a common `location_offset`, as well as `log_variance_samples`; this avoids
losing draw variation when all times share an extreme unit shift. The
`posterior_centered_location` and `posterior_location` fields expose the
posterior location mean in centered and absolute coordinates.

Only complete continuous positive event times are supported. Censoring,
rounded-time transforms, inferred prior defaults and native report parity remain
outside this scope. Workspace limits are checked before observation conversion
and random-number consumption. The posterior scale is retained logarithmically
to avoid exponentiating an extreme scale.

The implementation centers log times around a reference observation and uses
`log1p` for nearby relative times. All posterior arrays and the Johnson
diagnostic are included in the preflight storage bound. Posterior mean weights
are computed separately as `kappa0/kappa_n` and `n/kappa_n`; subtracting the
latter from one can erase a tiny weight with a still-material prior-location
contribution.

Independent base-R references in `tools/reference_lognormal_bayesian_gof.R`
cover conjugate posterior parameters, moments and Student-t predictive CDFs.
The joint diagnostic is integrated by conditioning on Gamma precision:
conditional Normal location intervals have constant observation bin patterns,
whose probabilities are integrated analytically, followed by one-dimensional
Gamma quadrature split at every change in interval ordering. Six cases include
concentrated priors, broad/equal times and time units changed by `1e-200` and
`1e200`. All 88 posterior, joint-draw, CDF and diagnostic summaries agree
within 2.850523 estimated Monte Carlo standard errors using 16,000 draws each.

The final integrated-code reference check took 0.0682 seconds after imports,
with 123.59 MiB peak resident memory and zero reported swaps. Two focused
tests cover conjugate/CDF identities, paired parameter use, unit invariance,
weak-prior location preservation and pre-RNG workspace rejection. Worker Ruff,
mypy and diff checks passed. No broad suite or new CI workflow was run.
