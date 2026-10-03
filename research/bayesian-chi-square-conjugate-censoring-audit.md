# Exponential and fixed-shape Weibull conjugate censoring audit

## Source contract

The cached BCSTTE User's Guide (`research/raw/BCSTTE/guide.pdf` in the source
checkout; SHA-256
`85ef8cb10d5043cebb89e1be770bf92317f107f3cf1bf41ace79dbb510f29bdc`) states
in §1 that input rows contain a possibly censored survival time and an event
indicator, with 1 for an event and 0 for right censoring. It also states that
the native program requires at least one uncensored observation. Section 4.1
gives the exponential density with mean/scale `eta`; §4.7 gives the Weibull
density as its hazard times survival:

```text
f(t | lambda, beta) = beta * lambda * t**(beta - 1) * exp(-lambda*t**beta)
lambda = eta**(-beta)
```

For independent, noninformative right censoring, an exact event contributes
`lambda * beta * t**(beta-1) * exp(-lambda*t**beta)`, while a censor contributes
`exp(-lambda*t**beta)`. Terms independent of `lambda` do not affect its
posterior. Thus, for a Gamma(`a`, rate=`b`) prior,

```text
lambda | t, event ~ Gamma(a + d, rate=b + sum(t**beta))
```

where `d` is the event count. Exponential is the `beta=1` case. This algebra
does not recover the native prior, fitting defaults, numerical fallbacks, or
censored-data goodness-of-fit transform.

## Python scope

`exponential_bayesian_gof` and `weibull_fixed_shape_bayesian_gof` accept an
optional actual-Boolean `event` vector; omission retains the previous all-event
contract and seeded behavior. Censored observations contribute their full
exposure to the posterior rate but do not increment posterior shape. Event
times must be positive; zero-time censors are valid and contribute zero
exposure. The diagnostic remains available only for complete observations;
censored data return `diagnostic=None`, because the cached guide does not
specify a censored PIT or imputation rule for Johnson's statistic.

The conjugate update is mathematically proper whenever posterior shape and
rate are positive. Consequently, these Python functions permit all-censored
data when the explicit Gamma prior yields a proper posterior. This is an
intentional extension beyond the native program's stated minimum-one-event
input rule, not a claim of native parity. The censoring mechanism is assumed
independent/noninformative for the event-time parameters.

Prior shape/rate may remain zero when the observed events and exposure produce
a proper posterior. All-zero censored data therefore require positive prior
shape and rate. Existing common time-scale centering and fixed-shape Weibull
power scaling are retained, and the CDF/diagnostic arrays are avoided when a
censored diagnostic is not defined.

## Integrated validation

Root ran 48 focused tests across the conjugate fits, unknown-shape Weibull,
the four additional TTE families, and success calibration: all passed with
warnings treated as errors. The test process used 2.609 seconds, peaked at
203.42 MiB RSS, and reported zero process swaps. Targeted lint/format checks
and mypy over all five changed statistical modules passed. These were focused
local checks; no full local suite, dependency installation or new CI workflow
was added.
