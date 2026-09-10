# Bayesian hierarchical binomial model (BHM-BLN)

Catalog entry **106** is implemented from the official
[BHM-BLN manual](https://biostatistics.mdanderson.org/shinyapps/BHM-BLN/BHM_BLN.pdf),
by Yanhong Zhou and J. Jack Lee. Its statistical comparisons are independent beta
posteriors, a pooled beta posterior, group-specific logistic-normal hierarchical
posteriors, and the transformed global logit under that hierarchy. The Python
implementation supplies those distributions or retained simulation draws for
analysis and plotting. [Source provenance](hierarchical-binomial-source.json)
records the inspected manual and live input defaults. Historical file templates,
report layouts and JAGS random-number sequences are not reproduced.

## Model and computation

```text
y_i | theta_i ~ Binomial(n_i, expit(theta_i))
theta_i | mu,tau ~ Normal(mu, 1/tau)
mu ~ Normal(prior_mean, 1/prior_mean_precision)
tau ~ Gamma(precision_shape, rate=precision_rate)
```

Gamma parameters use **shape/rate**, so the prior mean precision is shape/rate.
The model is not a beta-binomial hierarchy. An independent beta prior is used
only for the no-borrowing and complete-pooling comparison models.

Group logits use independent conditional
[elliptical slice updates](https://proceedings.mlr.press/v9/murray10a.html),
vectorized across groups and chains. They follow Gaussian-prior ellipses and
shrink an angular bracket until a proposal meets the likelihood slice. Binomial
log likelihoods use `logaddexp` to handle probabilities near zero and one without
subtracting rounded probabilities. Hyperparameter updates are Gibbs draws:

```text
mu | theta,tau:
  precision = prior_mean_precision + number_of_groups * tau
  mean = (prior_mean_precision * prior_mean + tau * sum(theta)) / precision

tau | theta,mu:
  shape = precision_shape + number_of_groups/2
  rate = precision_rate + sum((theta_i-mu)**2)/2
```

This implementation introduces no JAGS dependency and no proposal-width tuning
parameter. It still produces correlated MCMC draws, not independent or exact
posterior samples. Failure to obtain a slice draw after 1,000 bracket updates,
or an unrepresentable numerical state, raises `ArithmeticError`.

## Use and comparison models

```python
import numpy as np
from mdanderson_stats import hierarchical_binomial, summarize_chains

fit = hierarchical_binomial(
    successes=[2, 3, 5, 1, 2],
    trials=[10, 10, 10, 5, 3],
    draws=4000,
    warmup=1000,
    chains=4,
    rng=np.random.default_rng(106),
)
groups = summarize_chains(fit.group_probability)
overall = summarize_chains(fit.overall_probability)
logit_diagnostics = summarize_chains(fit.group_logit)
precision_diagnostics = summarize_chains(fit.precision)

independent_regions = fit.independent.credible_set(0.8)
pooled_region = fit.pooled.credible_set(0.8)
```

The retained arrays have axes `(chain, draw, group)` for group logits and
`(chain, draw)` for the global logit and precision. Probability properties apply
`expit` to the corresponding logits. The source's overall target is expit(mu),
which is neither a sample-size-weighted group average nor the mean probability
of a new random group. Raw logits remain available when transformed probabilities
round to endpoints.

Input vectors contain 2 through 100 groups, with nonnegative integer counts,
successes <= trials and total trial count below 2**53. Zero-observation groups
are allowed as prior-predictive groups. The caller supplies a NumPy Generator;
seeds are not silently reset. `draws` is the number **retained per chain**,
excluding `warmup`. At least two chains and eight retained draws are required.
Stored inputs, raw draws and comparison-model parameters are owned and read-only.
A failure after sampling begins may consume RNG state.

Current live defaults are prior mean zero, prior mean precision 1e-6, gamma
precision shape/rate 0.01/0.01, and comparison beta shapes 0.5/0.5. The manual's
worked comparison example instead uses beta shapes 0.001/0.001. Set
`independent_prior=(0.001, 0.001)` to reproduce its exact pooled shapes 13.001 and
25.001; its printed 13 and 25 are rounded.

## Summaries and diagnostics

`summarize_chains` returns pooled mean, median, standard deviation, an empirical
shortest interval of default mass 0.8, classical split R-hat, and a batch-means
Monte Carlo standard-error estimate. The interval uses the shortest sorted window
containing ceil(probability * total_draws) samples. It is an empirical connected
interval, not a claim to identify disconnected highest-density regions.

Split R-hat compares within- and between-half-chain variation; it is **not** the
modern rank-normalized statistic. Constant identical chains have undefined R-hat,
and constant separated chains have infinite R-hat. Batch-means MCSE uses batches
of approximately sqrt(draws) observations per chain and is only an asymptotic
estimate. Neither diagnostic proves convergence or resolves all mixing problems.
Inspect logits and precision as well as probability summaries, and retain draws
for stronger diagnostics in an established MCMC analysis tool. The API does not
silently label a run converged or equate iteration count with effective samples.
Weak precision priors can require substantially longer chains.

## Focused validation

Six tests verify analytic independent/pooled posteriors, seeded reproducibility,
input validation and retained-array ownership. Hierarchical group posterior means
are checked against independent one-dimensional numerical integration in a
concentrated-hyperprior limit. Additional checks recover conditional Gaussian
prior moments and the full joint hierarchy's prior moments with no data,
including Var(theta_i)=Var(mu)+E(1/tau). Summary checks cover empirical interval
mass and detection of deliberately separated chains. These establish numerical
and algorithmic behavior for the tested regimes; they do not guarantee convergence
for every user dataset or prior.
