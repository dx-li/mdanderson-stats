# Unknown-shape Weibull posterior with right censoring

The cached BCSTTE guide, §4.7, defines the Weibull density with shape `beta`
and scale `eta` as `beta/eta * (t/eta)^(beta-1) * exp(-(t/eta)^beta)`.
The guide does not specify a prior or executable fitting procedure. The Python
workflow therefore requires an explicit proper bivariate Gaussian prior on
`(log(beta), log(eta))`; it does not claim BCSTTE prior or software parity.

The implementation uses serial elliptical slice sampling in those Gaussian
coordinates. Times and log-scale are centered by the first positive follow-up
time for numerical stability. `parameters[..., 1]` is the centered log-scale
coordinate; `log_scale_offset` recovers the absolute coordinate. An actual
Boolean event indicator selects the source-defined Weibull density for exact
events and survival for right-censored observations. This is the ordinary
noninformative-censoring likelihood; no censoring mechanism is modeled. Zero
time is valid for a censor and contributes `S(0)=1`; event times must be
positive. A proper supplied prior also permits all-censored data.

For complete data, every posterior draw remains paired across observed-time
CDF evaluations and the existing Johnson posterior chi-square diagnostic is
applied. The guide does not define a censored-data CDF transform, so censored
fits return no diagnostic. Rounded-time models, native prior mapping, and
model-selection claims remain outside this workflow.

The sampling likelihood omits event-only parameter-independent log-time
constants to preserve time-unit invariance; retained absolute likelihoods
restore them. Under a time-unit multiplier, the reported log likelihood shifts
by minus the number of exact events times the log multiplier.
Preflight bounds include paired raw/frozen draws, CDF/count arrays and chain
summary temporaries. Runtime limits bound slice evaluations and data work.

`tools/reference_weibull_unknown_shape_gof.R` independently integrates two
fully bivariate posteriors with correlated Gaussian priors using tensor
Gauss-Legendre quadrature. Increasing order from 181 to 241 and widening the
domain from ±8 to ±10 prior standard deviations changes moments/CDFs by at
most `9.4689e-10`. The reference uses the guide's stable log-density equation
because the installed R Weibull density produces nonfinite intermediate
values in remote quadrature tails. Native R `pweibull` supplies the CDF.

`tools/reference_weibull_unknown_shape_censor.R` adds base-R quadrature
references for a mixed sample (including a zero-time censor) and an
all-right-censored sample. The two-dimensional rule increases from order 161
over ±8 to 201 over ±10 prior standard deviations; the largest change across
all posterior moments and paired CDF means is `7.19220239e-11`.
`tools/check_weibull_unknown_shape_censor.py` compares 21 summaries from
four-chain Python fits to those references. Maximum discrepancy is 1.501
batch-means MCSE and maximum split R-hat is 1.001993. The comparison takes
1.017 seconds after imports, peaks at 131.77 MiB and reports zero swaps.

Across 23 joint posterior mean/variance/covariance/CDF summaries, four-chain
Python draws agree within 2.280891 batch-means Monte Carlo errors; the maximum
classical split R-hat is 1.001747. The complete-data comparison takes 1.032
seconds after imports, peaks at 120.91 MiB and reports no swaps. The two
affected focused test files pass 14 checks; Ruff and format checks pass. The
unknown-shape module passes targeted mypy. No new CI workflow or broad suite
was introduced.
