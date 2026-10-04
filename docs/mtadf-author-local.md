# MTADF author-reference local logistic policy

The local author-reference functions implement the adjacent-dose logistic
policy in the authors' `targetAgentDF.r`, separately from the paper-based
`mtadf_local_logistic_decision` API.

```python
import numpy as np
from mdanderson_stats.mtadf_author_local import mtadf_author_local_decision

result = mtadf_author_local_decision(
    subjects=[6, 6, 0, 0],
    toxicities=[0, 1, 0, 0],
    responses=[4, 3, 0, 0],
    current_dose=1,
    rng=np.random.default_rng(2718),
    draws=1000,
    warmup=500,
    chains=4,
)
print(result.action, result.dose, result.probability_backward_nonpositive)
```

For dose levels `x = 1, ..., J`, the author standardizes the entire dose grid
as `xs = (x - mean(x)) / (2 * sample_sd(x))`. Each fitted window contains two
adjacent levels and uses `logit(p) = beta0 + beta1 * xs`, with independent
Cauchy priors of scales 10 and 2.5. Counts are used through the equivalent
aggregated binomial likelihood. An unobserved neighboring dose is allowed in
the fit; it contributes no likelihood. The public
`mtadf_author_local_posterior(..., lower_dose=i)` returns the established
`MTADFLocalLogisticPosterior` type, whose `window_doses` are these standardized
coordinates, not supplied dose concentrations. It requires at least one
observed subject in the pair.

The default author gates are `ce1=0.3`, `ce2=0.4`; the function validates
`0 <= ce1 <= ce2 <= 1`. Let `q_back = P(beta1 <= 0)` for the previous/current
pair and `p_forward = P(beta1 > 0)` for the current/next pair:

- At the lowest dose, escalate only when `p_forward > ce1`.
- At the highest dose, de-escalate only when `q_back > 1 - ce1`.
- At an interior dose whose next dose is untried, de-escalate if
  `q_back > 1 - ce1`, escalate if `q_back <= 1 - ce2`, and otherwise stay.
- If the next dose has observations, fit the forward pair first and the
  backward pair second. Escalate if `q_back <= 1 - ce2` and
  `p_forward > ce1`; otherwise de-escalate if `q_back > 1 - ce1`; otherwise
  stay.

All comparisons are strict or inclusive exactly as written. Movement is
clamped to the safety-admissible prefix. Safety uses the fixed prior and
inclusive cutoff implemented by `mtadf_author`; if only the lowest dose is
admissible, the author rule returns it without fitting efficacy. Before
enrollment, this API starts at dose index zero. `final=True` delegates to the
author all-dose `response / (subjects + 0.0001)` isotonic selection and does
not fit a local slope.

The sampler is the package's explicit bounded Python random-walk MCMC, with
caller-controlled seed, draws, warmup and chains. It is not the native
`mcmc::metrop` random stream or default proposal behavior. The author's
operating-characteristic routine also has a forced first-dose/second-dose
ramp and uses the cap computed before the latest cohort for movement; direct
decisions here use the fresh cap unless the separate internal simulator
helper is used. See the [source audit](../research/mtadf-author-local-audit.md).

Each fitted window is available on `forward_posterior` and
`backward_posterior`, including its acceptance, split-R-hat, and MCSE
diagnostics. Sparse adjacent-pair data can mix slowly: inspect those fields
and increase the draw budget when needed. The decision uses the finite-chain
posterior point probabilities returned by the sampler; the diagnostics do not
certify convergence.
