# Log-odds-rate posterior workflow audit

## Source contract

The cached BCSTTE guide §4.6 names the log-odds-rate family and supplies its
shape, scale, limiting cases, and moment statements. The displayed survival
formula omits `c` inside its bracket. Shen and Thall's primary generalized
odds-rate model gives

```text
S(t; scale, shape, c) = [1 + c * (t/scale)**shape]**(-1/c),
scale > 0, shape > 0, c > 0.
```

They identify `c=1` as log-logistic and the `c -> 0` limit as Weibull. The
article's covariate extension multiplies the power term by `exp(b'Z)`. Source
provenance and the guide/article distinction are recorded in the
[core source audit](log-odds-rate-source-audit.md), which cites Shen and Thall,
*Statistics in Medicine* 17 (1998), §2, equation (7), pages 1001–1002.

Let `d = log(t/scale)`, `h = shape*d`, and
`q = log(1 + c*exp(h))/c`. The sourced density and survival are

```text
log(S(t)) = -q
log(f(t)) = log(shape) - log(scale) + (shape-1)*d - q - log(1+c*exp(h))
F(t) = 1 - exp(-q).
```

The implementation uses absolute Gaussian-prior coordinates
`(log_shape, log_scale, log_c)`, centers log scale by a data-derived time
reference, and samples with the same elliptical-slice procedure as the two-
parameter TTE fits. Density calculations are for the centered time unit; each
event log density receives `-log(time_reference)` in the reported absolute log
likelihood. A censored survival term has no density Jacobian. One paired
posterior draw evaluates all complete-observation CDFs. Censored fits return
no Johnson diagnostic because the source does not specify a censored-data
transform.

The proper correlated Gaussian prior is a caller-selected Python contract.
Neither that prior nor the elliptical-slice fitter is attributed to BCSTTE.
The workflow does not implement the guide's moments; the primary survival
implies moment order `r` exists only when `shape > r*c`, so the finite-variance
condition is stronger than the finite-mean condition.

## Validation reference

`tools/reference_log_odds_rate.R` is an independent base-R calculation of the
source survival, density, CDF, and posterior summaries. Its explicit proper
correlated normal prior has mean `log([1.6, 1.2, 0.7])` and covariance

```text
[[ 0.130,  0.035, -0.018],
 [ 0.035,  0.190,  0.040],
 [-0.018,  0.040,  0.160]].
```

The reference uses five times `[0.45, 0.8, 1.3, 2.1, 3.4]`, once with all
events and once with event indicators `[1, 0, 1, 1, 0]`. Tensor
Gauss–Legendre orders 25, 35, and 45 provide three quadrature resolutions.
The reference's log-scale prior coordinate is relative to the first time
`0.45`; the Python fit therefore shifts the absolute prior mean's scale
coordinate by `log(0.45)` before centering it internally. The scale covariance
and cross-covariances are unchanged by this translation.
The fixture records means, variances and covariances of the three log
parameters, transformed shape/scale/`c` means, and posterior mean CDFs at all
five observation times. A separate identity fixture checks centered log
density, log survival, and CDF at three times. The reference prior is only for
validation and does not supply a native default.

The focused Python tests cover `c=1`, an interior `c`, the small-`c` Weibull
limit, far-tail representability, invalid inputs, complete/censored API
contracts, and time-unit changes. The seeded Python-to-R comparison is
configured for four chains, 16,000 retained draws per chain, 1,000 warmup draws
and dispersed starts. It compares 17 posterior summaries per event pattern
against the order-45 R fixture (34 comparisons total), estimating Monte Carlo
error by chain batch means.

The first comparison is invalidated because the R reference transformed
standard Normal rows using `t(chol(covariance))`. R's `chol` returns upper
triangular `U` with `t(U) U = covariance`; row-vector draws must multiply `U`,
not `t(U)`. The mistaken transform therefore used a different covariance.
The reference now asserts `crossprod(U) == covariance` to numerical tolerance,
multiplies row nodes by `U`, and the posterior fixture has been regenerated.
Order-35 to order-45 quadrature changes are at most `1.70e-6` (complete) and
`9.36e-7` (right-censored). No Python-versus-R posterior agreement claim is
made until the corrected fixture is checked with the same fixed seeds
(`20261003` complete, `20261004` right-censored). The comparison utility
records runtime, peak memory, MCSE-scaled errors, and split R-hat. No native
BCSTTE numerical parity is claimed.
