# Right-censored lognormal Bayesian fitting

`lognormal_right_censored_bayesian_fit` fits mixed exact-event and
right-censored times with a separate lognormal workflow. It uses the same
explicit proper Normal-Inverse-Gamma prior as the
[complete-data workflow](lognormal-bayesian-gof.md), then samples censored
log-times as latent values in a serial Gibbs chain. The complete-data API and
its exact independent-draw behavior are unchanged.

```python
import numpy as np
from mdanderson_stats import lognormal_right_censored_bayesian_fit

fit = lognormal_right_censored_bayesian_fit(
    [0.4, 1.2, 1.2, 2.6, 5.0, 0.0, 3.5],
    event=[True, False, False, True, False, False, False],
    prior_location=0.3,
    prior_location_precision=2.0,
    prior_variance_shape=4.5,
    prior_variance_scale=1.1,
    draws=2000,
    warmup=1500,
    chains=4,
    rng=np.random.default_rng(6609),
)
assert fit.centered_location_samples.shape == (4, 2000)
assert fit.parameter_summary.batch_mean_mcse.shape == (2,)
assert not fit.event[5]  # zero-time censor is retained in metadata
```

## Model and posterior update

Let `Y=log(T)` and `V=sigma**2`. The prior is
`V ~ InverseGamma(a0,b0)` and `mu | V ~ Normal(m0,V/kappa0)`, with all prior
parameters positive. Exact events contribute the lognormal density. A positive
right censor at `c` contributes
`S(c | mu,V) = Phi((mu-log(c))/sqrt(V))`. Under independent, noninformative
censoring, the posterior after integrating censored rows is not another
Normal-Inverse-Gamma distribution.

The sampler augments each positive censor with its latent `log(T)` from the
Normal distribution truncated below `log(c)`. It then draws the variance and
location from the usual NIG conditional based on event log-times and those
latent values. The implementation uses stable normal-tail likelihoods and a
bounded exact truncated-normal sampler: ordinary-normal rejection when the
standardized threshold is nonpositive, and an exponential-tail rejection
sampler when it is positive. Proposal exhaustion, unrepresentable tails, and
arithmetic failures raise contextual errors; values are never clipped to a
censoring threshold.

`centered_location_samples` and `location_offset` encode absolute log-location
as their sum. `log_variance_samples` remain paired with those location draws.
`parameter_summary` reports pooled means, medians, standard deviations,
shortest intervals, split R-hat, and batch-means MCSE for centered location and
log variance. These estimates do not guarantee convergence. The retained
`log_likelihood_samples` are observed-data log likelihoods conditional on
each paired parameter draw, including the event-time density Jacobian.
`likelihood_evaluations`, `likelihood_work_units`, `gibbs_updates`,
`gibbs_work_units`, `parameter_draw_work_units`, `total_work_units`, and
`truncated_normal_proposals` expose the bounded-work accounting. `max_work`
counts augmentation/update work, observed-likelihood rows, parameter draws, and
truncated-normal proposals; both configured proposal caps and minimum
per-iteration work are checked before RNG use.

Zero-time censors are preserved in `times` and `event` but contribute exactly
`S(0)=1`; they are excluded from latent sampling and posterior updates. If every
row is a zero-time censor, outputs are exact independent draws from the prior.
If positive events are present and the only censors are at zero, the function
uses exact independent complete-data posterior draws. In these exact-draw
cases `warmup` is not used and the returned `warmup` is zero. All-censored data
with positive censoring times are supported under the proper prior. This is a
Python extension beyond the BCSTTE guide's stated
minimum-one-event input rule. All-complete input is rejected with a pointer to
the exact independent-draw `lognormal_complete_data_bayesian_gof` workflow.

No Johnson goodness-of-fit diagnostic is returned for censored data: the
available source does not define a censored PIT or imputation rule. The prior
and sampler are explicit Python choices, not recovered native fitting
defaults.

Changing time units by factor `u` requires adding `log(u)` to `prior_location`;
the other prior parameters are unchanged. Centered location and log-variance
draws should then agree within floating-point precision.

## Source and audit

The BCSTTE guide §1 defines event versus right-censor indicators, and §4.5
defines `log(T) ~ Normal(mu,sigma**2)` with `sigma` as log-time standard
deviation. It does not specify a censored fitting algorithm, prior, or
censored-data goodness-of-fit transform. The equations and validation scope are
recorded in the [source audit](../research/lognormal-right-censored-audit.md).
