# CiBolus prior calibration scope

The implementation follows the three-stage prior elicitation described in
Thall et al. (2011), §4.2: elicit regimen-specific response/toxicity
probabilities; treat a supplied full joint category grid as the pseudo-data
truth and average fitted pseudo-posterior log-coordinate means; then inspect
prior probability moments and beta-moment ESS. The implementation accepts the
full response-category × toxicity probability grid so it does not invent an
interpolation between response-time elicitation points. Each pseudo-dataset is
balanced with the same number of patients at every concentration/bolus
regimen. The returned prior SD is caller-supplied: the paper's illustrative
setting uses 50 patients at each of eight regimens (400 total), a diffuse
pseudo-prior with log-coordinate SD 20, and final log-coordinate SD 9 (variance
81); these are reference context, not API defaults or universal calibration
recommendations.

`cibolus_prior_predictive_moments` reports empirical means and population
variances (ddof=0) for joint cells, p0, cumulative response at all supplied endpoints,
toxicity conditional on bolus response, toxicity at response time one, and
toxicity after failure. The explicit source ESS subset is p0, response by time
one, toxicity at response time zero, and toxicity at response time one. It
uses `mean*(1-mean)/population_variance - 1`; constant interior metrics have
limiting ESS +infinity, while constant endpoint metrics have undefined ESS.
Failure toxicity is reported as an additional diagnostic, not silently folded
into the paper's response-time-one metric. No aggregate ESS over unrelated
probabilities is returned, and no variance optimization is performed.

Pseudo-posterior fitting reuses the serial CiBolus fitter and its diagnostics.
The likelihood input ceiling was raised from 200 to 400 observations in both
the public likelihood validator and fitter to support the paper's 400-record
pseudo-study; the complete-outcome trial patient cap remains independent.
Repetitions, prior, Monte Carlo settings, seeds, and cumulative evaluation/work
limits are explicit. The procedure is a Python implementation of the described
pseudo-data workflow, not native executable or prior-mean parity: the paper
uses an unavailable mode-initialized Gibbs implementation, while this package
uses elliptical slice sampling. Its output includes between-pseudo-replicate
uncertainty and per-fit sampler diagnostics to make that distinction visible.

A bounded source-sized smoke used the four-by-two regimen grid, 50 patients
per regimen (400 total), an explicit all-zero log-mean / SD-20 pseudo-prior,
one pseudo-replicate, two chains, eight retained draws, and 100,000 evaluation /
5,000,000 work-unit caps. It completed in 2.01 seconds with 85 likelihood
evaluations and 34,512 work units (124,567,552-byte peak RSS, zero swaps).
The maximum split-Rhat was 4.11, so this confirms bounded 400-record execution
only; eight draws are far too few to assess convergence or claim an elicited
prior estimate.

The retained-cell preflights include the raw/frozen fit arrays, conservative
`14 * chains * draws * (11 + regimen_count)` chain-summary scratch, and live
probability/count/output buffers. Probability moment calculations clip only
64-epsilon excursions outside [0,1] and reject larger violations; materially
negative beta-moment ESS is also an arithmetic error rather than a reported
invalid value.

Bolus and cumulative response probabilities are evaluated directly from the
response CDF for each retained log-parameter draw. They are not reconstructed
by summing joint response/toxicity cells, which can introduce artificial
response variance when only toxicity coordinates vary. The extra CDF evaluations
are included in work and live-cell preflights.

## Independent numerical reference

`tools/reference_cibolus_calibration.py` regenerates two balanced pseudo samples
with 20 observations each, using seed 884821. Only log(beta0) varies, with prior
mean -2.5 and SD 0.7; the other coordinates are fixed. Fits use two chains,
600 retained draws and 200 warmup iterations. The actual generated counts,
settings and Python summaries are preserved in
`tests/fixtures/cibolus-calibration/`.

`tools/reference_cibolus_calibration.R`, given that directory as its argument,
integrates the reduced posterior independently using those exact counts. The
response likelihood is constant in this coordinate and cancels. The two fitted
log(beta0) means differ from quadrature by 0.342 and 0.552 reported sampler
Monte Carlo standard errors. This checks a numerical slice of the workflow,
not the paper's full eleven-coordinate calibration.

The same reference integrates prior response and toxicity moments. At 5,000
direct prior draws (seed 884822), the four varying toxicity means differ by at
most 0.035 Monte Carlo standard errors. Sampled toxicity variances differ from
the quadrature population variances by at most 1.987%; beta ESS follows the
reported sampled moments. Fixed response means agree within 4.45e-16 and now
have exactly zero variance and infinite ESS. That last comparison exposed and
motivated the direct-response-CDF correction above.

The corrected root generator run took 18.560 seconds, peaked at 119.91 MiB
RSS and reported zero swaps. Its generated counts and posterior summaries
are unchanged from the earlier independent comparison. Five focused calibration
checks pass after the correction; targeted Ruff and mypy checks pass. No broad
CI or full package numerical run was added.
