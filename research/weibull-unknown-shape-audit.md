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
