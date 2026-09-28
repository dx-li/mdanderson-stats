# Unknown-shape Weibull complete-data posterior

The cached BCSTTE guide, §4.7, defines the Weibull density with shape `beta`
and scale `eta` as `beta/eta * (t/eta)^(beta-1) * exp(-(t/eta)^beta)`.
The guide does not specify a prior or executable fitting procedure. The Python
workflow therefore requires an explicit proper bivariate Gaussian prior on
`(log(beta), log(eta))`; it does not claim BCSTTE prior or software parity.

The implementation uses serial elliptical slice sampling in those Gaussian
coordinates. Times and log-scale are centered by the first observed time for
numerical stability. `parameters[..., 1]` is the centered log-scale coordinate;
`log_scale_offset` recovers the absolute coordinate. Every posterior draw is
kept paired across all observed-time CDF evaluations, then the existing Johnson
posterior chi-square diagnostic is applied. Inputs are complete positive event
times; censoring, rounded-time models, native prior mapping, and model-selection
claims are outside this workflow.

The sampling likelihood omits the parameter-independent log-time constant to
preserve time-unit invariance; retained absolute likelihoods restore it.
Preflight bounds include paired raw/frozen draws, CDF/count arrays and chain
summary temporaries. Runtime limits bound slice evaluations and data work.

`tools/reference_weibull_unknown_shape_gof.R` independently integrates two
fully bivariate posteriors with correlated Gaussian priors using tensor
Gauss-Legendre quadrature. Increasing order from 181 to 241 and widening the
domain from ±8 to ±10 prior standard deviations changes moments/CDFs by at
most `9.4689e-10`. The reference uses the guide's stable log-density equation
because the installed R Weibull density produces nonfinite intermediate
values in remote quadrature tails. Native R `pweibull` supplies the CDF.

Across 23 joint posterior mean/variance/covariance/CDF summaries, four-chain
Python draws agree within 2.280891 batch-means Monte Carlo errors; the maximum
classical split R-hat is 1.001747. The comparison takes 1.032 seconds after
imports, peaks at 120.91 MiB and reports no swaps. Three focused checks pass
in 2.62 seconds, along with Ruff, formatting and mypy. No new CI workflow or
broad suite was introduced.
