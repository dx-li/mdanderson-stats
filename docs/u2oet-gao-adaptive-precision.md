# Adaptive corner precision for GAO fits

`fit_u2oet_gao_adaptive_precision` applies the U2OET guide's four-corner
utility precision target to the existing 2017 GAO sampler. It requires explicit
normal-prior coordinates, utility, target, retained-draw start and maximum;
it does not infer native defaults or claim native executable parity.

```python
import numpy as np

from mdanderson_stats import (
    fit_u2oet_gao_adaptive_precision,
    u2oet_gao_parameter_names,
)

names = u2oet_gao_parameter_names(2, 2)
prior_mean = np.array(
    [-1.0, -0.2, 0.4, 0.7, 0.0, -0.2, 0.3, 0.2, 0.3, 0.0, np.log(1.2), 0.0]
)
prior_sd = np.zeros(len(names))
prior_sd[0] = 0.7
counts = np.zeros((2, 2, 2, 2))
counts[0, 0, 1, 0] = 3
counts[0, 0, 0, 0] = 1
toxicity_only = np.zeros((2, 2, 2))
toxicity_only[1, 1, 1] = 2
result = fit_u2oet_gao_adaptive_precision(
    [1, 2],
    [1, 2],
    counts,
    toxicity_only=toxicity_only,
    prior_mean=prior_mean,
    prior_sd=prior_sd,
    utility=[[0, 1], [2, 0]],
    target_mcse_ratio=0.05,
    initial_draws=8,
    max_draws_per_chain=32,
    batch_draws=8,
    warmup=8,
    chains=2,
    rng=np.random.default_rng(20261003),
)
print(result.target_met, result.termination, result.draws_per_chain)
print(result.mcse_ratio)  # chain rows; columns follow corner_indices
print(result.corner_split_rhat)  # reported separately from the stop rule
```

The guide specifies checking batch-means MCSE divided by posterior SD for each
of four dose-grid corner utilities, separately within each chain. Its
simulation target range is 0.001–0.05; it also says retained draws per chain
are at least the burn-in amount. The guide does not specify batch length,
extension size, maximum work, or restart protocol. This implementation uses
the existing GAO fitter's 2–16 chain range. It performs warmup once and
continues each chain from its complete last retained coordinate vector,
including association Fisher-z. The current GAO transition has no adaptation or
latent auxiliary state to carry between chunks.

Python uses nonoverlapping batch means of length
`max(2, floor(sqrt(draws)))`; any incomplete trailing batch is excluded only
from the MCSE calculation. The SD and retained fit include every draw. Each
extension is at most `batch_draws`, except that a final chunk may absorb up to
seven leftover draws to meet the sampler's minimum eight-draw call. If fewer
than eight draws remain under the cap, no additional fitter call is made and
the returned count may be below that upper bound. A zero-SD corner has
undefined precision and does not pass the
target. A `draw_cap` result means at least one chain/corner failed the requested
ratio before the explicit upper bound.

The returned `fit` is a regular `U2OETGAOFit`; its joint probabilities can be
flattened over chain and draw axes for existing posterior utility functions.
Split-Rhat is reported separately and is not the stopping rule. Precision at
the four corners does not establish precision at other dose combinations or
for every model parameter. Chunk boundaries change random-number consumption,
so this workflow is not expected to match a single fixed-length fit from the
same seed.

Before consuming randomness, the wrapper checks aggregate worst-case
likelihood evaluations and full-grid work over warmup and every possible
chunk, including one deterministic initial-state likelihood check per chain,
plus retained and temporary array estimates. These conservative limits
reuse the GAO sampler's per-evaluation hard caps and the adaptive wrapper's
4-million retained-joint-cell and 12-million live-cell policies. They are
resource estimates, not RSS or runtime guarantees.
