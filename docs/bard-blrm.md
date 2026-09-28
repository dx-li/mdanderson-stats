# BARD paper Bayesian logistic toxicity model

The BF-BLRM component evaluates and fits the dose-toxicity model in the
[BARD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240483/). Its parameters,
prior and target interval are explicit. The current BARD app guide describes
BF-BOIN stage one; this component implements the paper's alternative model
and does not claim equivalence to a hidden app backend.

The printed model is

```text
logit(p_j) = log(alpha) + beta * (dose_j / reference_dose)
alpha > 0, beta > 0
```

This is a regression on the **raw dose ratio**. Doses and the reference must
use the same units. `BARDLogisticPrior` supplies independent normal priors on
`(log(alpha), log(beta))`, in that order. Its second argument contains standard
deviations, not variances. A zero standard deviation explicitly fixes that
coordinate at the supplied mean.

```python
import numpy as np
from mdanderson_stats import (
    BARDLogisticPrior,
    bard_blrm_probability,
    fit_bard_blrm,
)

doses = [0.5, 1, 2]
curve = bard_blrm_probability(doses, 1, log_alpha=-1.5, log_beta=-1.2)
assert np.all(np.diff(curve) > 0)

fit = fit_bard_blrm(
    doses,
    patients=[4, 6, 4],
    toxicities=[0, 2, 3],
    reference_dose=1,
    prior=BARDLogisticPrior(mean=[-1.5, -1.2], standard_deviation=[0.8, 0.5]),
    target_interval=[0.16, 0.33],
    draws=64,
    warmup=32,
    chains=2,
    rng=np.random.default_rng(165),
)
assert fit.probability_draws.shape == (2, 64, 3)
print(fit.posterior_target_probability)
print(fit.posterior_overdose_probability)
```

The short chains demonstrate the API; choose sampling settings for the
required precision and inspect the diagnostics. The illustrated prior is not
a calibrated prior or a native default. The paper's simulation uses different
prior values, which are recorded in the [source audit](../research/bard-blrm-audit.md).

## Data and output

Inputs are grouped evaluable patient and toxicity counts at each dose.
Include observations from escalation and backfill patients together; pending
outcomes do not count as nontoxic observations. Dose values must be positive
and strictly increasing. The fit predicts at those same dose values, including
doses with zero observations.

For the supplied interval `(gamma1, gamma2)`, the outputs estimate

- `posterior_target_probability`: `Pr(gamma1 < p_j < gamma2 | data)`;
- `posterior_overdose_probability`: `Pr(p_j >= gamma2 | data)`.

Comparisons use log odds to preserve strict and inclusive boundaries when
ordinary probabilities round to zero or one. Arrays retain `(chain, draw,
dose)` axes for risks and indicators, and `(chain, draw, 2)` for log-parameters.
Coefficient, risk and both indicator summaries include split R-hat and
batch-means Monte Carlo errors. Constant coordinates can have undefined
R-hat; diagnostics are not convergence guarantees.

The sampler uses normal-prior elliptical slice updates, with independent
prior initializations for data-fitting chains. Without observations,
it draws directly from the prior. Fixed coordinates remain fixed. All retained
arrays are immutable. Explicit evaluation and work limits bound computation;
unrepresentable predictors or exhausted limits fail clearly. No numerical
failure silently substitutes a probability or changes the prior.
Limits are 100 dose levels, two million retained cells, two million likelihood
attempts and 50 million dose-level work units. Static limits are checked before
random draws; both likelihood attempts and retained predictions consume work.

## Validation and remaining scope

Independent base-R model evaluation and one/two-dimensional normal-prior
integration supply 20 probability and 22 posterior references. The posterior
examples cover a fixed slope and two free coordinates, including target and
overdose probabilities. The reference likelihood omits binomial coefficients;
its normalization constant is not a full binomial marginal likelihood for
model comparison. See [reference generator](../tools/reference_bard_blrm.R)
and [audit](../research/bard-blrm-audit.md) for comparison results and bounds.
Both four-chain posterior comparisons agree within two estimated Monte Carlo
standard errors, with maximum split R-hat below 1.002. This validates the
checked examples, not arbitrary prior/data configurations.

[BF-BLRM decision helpers](bard-blrm-decisions.md) provide one-step dose
movement, backfill eligibility and final MTD selection. The
[calendar replay](bard-blrm-trials.md) combines these with explicit arrivals,
potential outcomes and separate toxicity/response assessment delays.
Integration with [BARD stage two](bard.md) remains open; existing
[BF-BOIN](bf-boin.md) methods are also available.
