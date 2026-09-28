# Fixed-shape Weibull posterior diagnostic

The BCSTTE guide's §4.7 gives the Weibull shape/scale density and survival
function. Writing `lambda = scale**(-shape)` yields
`f(t|lambda,shape) = shape * lambda * t**(shape-1) * exp(-lambda*t**shape)`.
For caller-fixed positive shape and complete positive event times, a
Gamma(shape=`a`, rate=`b`) prior on lambda therefore updates exactly to
Gamma(`a+n`, rate=`b+sum(t**shape)`). The posterior rate prior is an explicit
Python contract, not a recovered BCSTTE default.

`weibull_fixed_shape_bayesian_gof` samples this exact one-dimensional posterior,
retaining the log rates in centered form plus an offset so extreme time scales do
not erase Gamma variation. It computes each observed time's Weibull CDF for the
same joint parameter draw and delegates to `bayesian_chi_square_cdf` (Johnson 2004). This follows the
paper's continuous-data posterior CDF statistic; shape is held fixed. The
Gamma prior may be improper at a zero shape or rate hyperparameter when the
resulting posterior remains proper after the required complete observations.

This method is limited to complete continuous times. It does not establish the
native BCSTTE prior, unknown-shape fitting, censored-data PIT/imputation,
rounded-time likelihood, numerical fallback, Rychlik rank convention, or native
report parity. The BCSTTE guide accepts right-censored data and mentions
rounding intervals, but does not describe the corresponding diagnostic
transformation or fitting algorithm.

The result exposes `centered_log_rate_samples` and `log_rate_offset`; their sum
is each sampled log rate. It also exposes `posterior_rate_log_scale` and
`posterior_rate_scaled_sum`, which encode the posterior denominator without
forming an extreme rate. The convenience scalar `log_posterior_rate` may lose
low-order precision at extreme scales.

Validation includes reduction to the existing exponential Gamma update at shape
one, a direct conjugate/CDF identity, time-unit rescaling, extreme-scale cases,
and pre-RNG workspace/representability failures.

An independent R calculation checked seven posterior cases and 76 summaries,
including centered log-rate moments, CDF means, and Johnson diagnostic
quantities; all discrepancies were within 1.671 Monte Carlo standard errors. The
checks included time-unit scales of 1e-200 and 1e200 and shape 1e305 with
identical event times. The latter uses centered relative powers so the finite
Gamma/n hazard variation remains visible.

The independent generator is `tools/reference_weibull_bayesian_gof.R`;
`tools/check_weibull_bayesian_gof.py` compares the two stored reference tables
with 16,000 independent posterior draws per case. Diagnostic expectations are
integrated over the finite intervals where all bin memberships are constant,
rather than estimated from a second Monte Carlo run. Reference tables retain
17 significant figures; the first comparison detected and corrected the R
CSV writer's default lower-precision serialization. The final comparison
takes 0.047 seconds after imports, peaks at 126.47 MiB and reports zero swaps.

Six focused checks pass in 1.42 seconds. Review corrected both common-offset
loss of Gamma variation and cancellation for adjacent representable times at
shape `1e305`. Storage preflight includes working vectors and chunk buffers
in addition to retained arrays, before large conversions or random draws.
Targeted Ruff, formatting, mypy and diff checks pass. No new numerical
dependency, CI workflow or broad numerical suite was added.
