# Original 2010 GAO cohort trials

The 2010 GAO decision uses posterior mean utility over every elementary joint outcome (paper Eq. 10–11). The interim trial first applies the global toxicity stop from Eq. 12: stop only when, at every dose pair, the posterior probability that the highest toxicity-category probability exceeds the prespecified limit is strictly greater than `stopping_probability`. If the trial continues, an interim allocation maximizes posterior mean utility among eligible pairs. Previously treated pairs and coordinatewise lower pairs remain eligible; untried escalation is restricted to the paper's three adjacent upper neighbors. The paper does not resolve a new mixed-direction move, so this implementation conservatively excludes untried mixed-direction pairs. A final Eq. 11 recommendation considers the full grid without interim movement restrictions.

The cohort simulator begins at the caller's explicit `starting` pair without a prior safety fit, then fits after each cohort (or the final patient-budget remainder). It accepts explicit categorical joint truth and an outcome-independent efficacy-evaluability probability. Unevaluable efficacy is represented as toxicity-only data. An optional two-uniform-per-patient tape makes generated outcomes replayable. This is a Python cohort workflow using the existing GAO posterior fitter; it does not reproduce the paper's native Gibbs sampler, prior elicitation, or within-patient clinical adaptation.

```python
import numpy as np

from mdanderson_stats.u2oet_gao2010_trial import simulate_u2oet_gao2010_trial

# Two efficacy and two toxicity categories at each pair; each row sums to one.
truth = np.full((2, 2, 2, 2), 0.25)
trial = simulate_u2oet_gao2010_trial(
    [0.0, 1.0],
    [0.0, 2.0],
    truth,
    [[0.0, 0.0], [1.0, 1.0]],
    prior_mean=np.zeros(12),
    prior_sd=np.zeros(12),
    starting=(0, 0),
    n_patients=6,
    cohort_size=3,
    efficacy_evaluability=0.9,
    toxicity_limit=0.5,
    draws=100,
    warmup=50,
    chains=2,
    rng=np.random.default_rng(2026),
)
print(trial.selected_pair, trial.stop_reason)
```

The simulator requires an explicit NumPy `Generator`. Work and retained arrays are bounded; cumulative fit budgets are checked before random state is consumed. Posterior split-Rhat and utility MCSE are descriptive diagnostics, not proof of convergence. Utilities are summarized after scale normalization to avoid overflow where possible; unrepresentable summaries raise an arithmetic error.
