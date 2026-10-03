# Log-normal right-censoring source and implementation audit

## Source contract

The cached BCSTTE User's Guide is indexed at
`https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BCSTTE/BCSTTE_UsersGuide.pdf`.
The local cached copy inspected in the source checkout is
`research/raw/BCSTTE/guide.pdf`, SHA-256
`85ef8cb10d5043cebb89e1be770bf92317f107f3cf1bf41ace79dbb510f29bdc`.
Guide §1 specifies each row's observed time and event/death flag (`1` for an
uncensored event, `0` for right censoring), and says the native program requires
at least one uncensored observation. Section 4.5 parameterizes `log(T)` as
Normal with mean `mu` and standard deviation `sigma`; therefore `V=sigma**2`
is the log-time variance.

The guide gives neither a prior nor a censored log-normal fitting algorithm.
The implementation therefore uses the existing Python convention:

```text
V ~ InverseGamma(a0, b0)             # shape / scale
mu | V ~ Normal(m0, V / kappa0)
```

All four hyperparameters are explicit, with `kappa0,a0,b0 > 0`. Under
independent, noninformative right censoring, an exact event at `t` contributes
the log-normal density, including the `-log(t)` Jacobian. A positive censor at
`c` contributes
`log(Phi((mu-log(c))/sqrt(V)))`. Thus the integrated censored posterior is not
the complete-data NIG posterior with a simple substitution of the event count.

## Gibbs contract

For each positive censored time, the sampler draws its latent `Y=log(T)` from
`Normal(mu,V)` truncated below `log(c)`. It combines those latent log-times
with exact-event log-times and updates the NIG conditional using the augmented
sample size `m`:

```text
kappa_n = kappa0 + m
m_n = (kappa0*m0 + sum(y))/kappa_n
a_n = a0 + m/2
b_n = b0 + sum((y-ybar)^2)/2
          + kappa0*m*(ybar-m0)^2/(2*kappa_n)
V | y ~ InverseGamma(a_n,b_n)
mu | V,y ~ Normal(m_n,V/kappa_n)
```

These are Gibbs conditionals for latent-completed data, not marginal
posterior hyperparameters. The positive-tail truncated Normal uses an exact
exponential proposal and rejection; nonpositive standardized bounds use
ordinary-normal rejection. A shared explicit proposal cap and a total-work cap
bound the sampler. Bounds outside the representable standardized domain,
threshold-rounding loss, proposal exhaustion, or other numerical failures raise
errors with chain/iteration/row context where applicable; draws are not
clipped to censoring thresholds.

## Edge cases and scope

Exact-event times must be positive; right-censor times may be zero. Zero-time
censors have survival likelihood one and are kept in result metadata but
excluded from latent augmentation and sufficient statistics. If all rows are
zero-time censors, the posterior equals the supplied prior and is sampled
independently, with zero observed-data log likelihood and no MCMC update. If
events are present but all censors are at zero, the exact complete-data NIG
posterior is drawn independently; warmup is unused and reported as zero. If
positive censors are present, all-censored data are supported because the
explicit prior is proper. This is a mathematical Python extension beyond the
native guide's minimum-one-event input rule. All-complete input is routed to
the existing exact independent-draw `lognormal_complete_data_bayesian_gof`
function; that API remains unchanged.

The new API returns paired centered-location and log-variance chains,
`ChainSummary` means/medians/standard deviations/intervals/split-R-hat/batch
MCSE, conditional observed-data log-likelihood draws, input event metadata,
work counters and proposal counts. It does not return a Johnson GOF diagnostic
or label the Gibbs conditionals as conjugate posterior hyperparameters. The
available source specifies no censored PIT or imputation rule.

Location centering uses the first positive observed time, with the prior mean
shifted by the same offset. A time-unit factor shifts only the prior log-time
location by `log(factor)`; inverse-gamma hyperparameters remain unchanged.
All-zero censor inputs use the prior location itself as offset. Preflight
validates dimensions, prior propriety, retained storage, iteration work and the
minimum proposal count before consuming the caller's RNG. Retention is limited
to parameter, observed-likelihood and input metadata arrays.

Independent validation compared posterior means, variances, covariance and
posterior mean CDFs against risk's separate base-R quadrature of the direct
observed-data likelihood, not another latent-augmentation implementation. It
used the fixed seeds `20261033` (mixed) and `20261034` (all censored), 12,000
retained draws per chain, four chains, and 3,000 warmup updates. All 23
case/metric comparisons were within 5 batch-means MCSE; the maximum was 2.0010
MCSE. Maximum split R-hat was 1.000292. The run took 9.68 seconds and peaked at
138,543,104 bytes (132.03 MiB) resident memory, with no swaps. The mixed case
used 301,902 truncated-normal proposals and 997,902 total work units; the
all-censored case used 371,391 proposals and 959,391 work units. See
`tools/check_lognormal_censored_posterior.py` for the fixed comparison
configuration. Risk's grid-121 to grid-151 quadrature change was at most
`9.65e-13` (mixed) and `2.08e-13` (all-censored); widening the order-151 domain
changed them by `3.86e-13` and `7.60e-10`, respectively.

No native prior, fitting-default, censored-Johnson, or report-parity claim is
made.
