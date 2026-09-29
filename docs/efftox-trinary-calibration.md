# Trinary EffTox prior elicitation

Trinary EffTox has mutually exclusive toxicity, efficacy without toxicity and
neither outcomes. Write `T` for toxicity probability and `Q` for efficacy
conditional on no toxicity. Marginal efficacy is `(1-T)*Q`. Two independent
Gaussian coefficient blocks describe the logistic dose relationships; both
slope distributions are conditioned to be positive. Their supplied Gaussian
means describe the distributions **before truncation**, and can be negative
when their standard deviations are positive.

`efftox_trinary_prior_moments` evaluates the induced probability means,
variances and moment-matched beta effective sample sizes. It integrates over
both independent prior blocks when calculating marginal efficacy uncertainty.

```python
import numpy as np
from mdanderson_stats import EffToxTrinaryPrior, efftox_trinary_prior_moments

prior = EffToxTrinaryPrior(
    mean=[-1.1, 0.8, 0.25, 1.2],
    sd=[0.45, 0, 0.6, 0],
)
moments = efftox_trinary_prior_moments([1, 2, 4], prior)
np.testing.assert_allclose(
    moments.efficacy_mean,
    (1 - moments.toxicity_mean) * moments.conditional_efficacy_mean,
)
assert np.all(moments.efficacy_variance > 0)
```

For a probability with prior mean `m` and variance `v`, beta-moment ESS is
`m*(1-m)/v - 1`. A fixed probability has infinite ESS. This measures prior
information, not effective MCMC sample size. The implementation uses stable
complements for rare events and the nonnegative product-variance expansion;
it avoids subtracting nearly equal second moments for concentrated priors.

## Calibrate elicited means and information

`calibrate_efftox_trinary_prior` fits toxicity first, then fits the conditional
efficacy block while retaining uncertainty from the fitted toxicity prior.
Efficacy inputs and its ESS target refer to **marginal efficacy**, not `Q`.
The recovered software sources do not specify a trinary calibration objective;
this sequential elicitation policy is an explicit Python choice. It is not
claimed to reproduce the original Windows calibration.

```python
from mdanderson_stats import calibrate_efftox_trinary_prior

fit = calibrate_efftox_trinary_prior(
    [1, 2, 4],
    efficacy_means=[0.15, 0.24, 0.34],
    toxicity_means=[0.08, 0.14, 0.22],
    efficacy_target_ess=0.8,
    toxicity_target_ess=1.1,
)
assert fit.toxicity_optimizer_success and fit.efficacy_optimizer_success
assert np.max(np.abs(fit.efficacy_mean_residual)) < 0.005
np.testing.assert_allclose(fit.moments.efficacy_effective_sample_size.mean(), 0.8, atol=0.001)
# fit.prior can be passed to the trinary posterior and trial simulation APIs.
```

Each stage minimizes the sum of squared mean residuals, plus `0.1` times the
squared difference between the average dose-specific ESS and its target, plus
`0.02` times the squared difference between the block's intercept and slope
standard deviations. Targets are therefore a fitting criterion, not guaranteed
exact matches. Inspect the achieved moments, residuals, optimizer success,
messages, evaluation counts and active parameter bounds before using a fit.
Separate outcome-specific ESS targets override the shared `target_ess` value.

Elicited toxicity means must be nondecreasing, and marginal efficacy must be
less than the fitted non-toxicity mean at each dose. Feasibility of these means
does not guarantee that every target is attainable under the logistic prior.
Physical doses use the same centered log-dose convention and optional zero-dose
shift as [EffTox](efftox.md).

## Numerical scope

The implementation reuses the established adaptive logistic-normal integration,
including positive-slope conditioning. Quadrature diagnostics are distinct from
optimizer convergence. Iteration, evaluation, parameter and deterministic work
limits bound calibration; a larger dose set or quadrature request can exceed
the work cap even with otherwise valid settings. The default evaluation cap is
2,000 per stage, with a 1,500-iteration cap. Computation runs serially.

Independent base-R fixtures cover twelve dose/prior combinations, including
nearly fixed coefficients. Focused checks cover fixed priors, extreme logits,
marginal variance and ESS, incompatible means, separate ESS targets and the
reported objective. The ordinary three-dose example above converges in both
stages with default settings. Native trinary calibration and desktop report
parity remain open; the [source audit](../research/efftox-trinary-calibration-audit.md)
records the distinction.
