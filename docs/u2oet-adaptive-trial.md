# Adaptive posterior precision in U2OET trial simulation

`simulate_u2oet_trial` keeps its fixed-budget behavior by default. To opt into
the guide's corner-utility precision check, supply explicit adaptive settings.
The simulation uses the existing separate data and posterior random streams,
and applies the precision target to every newly fitted sufficient-statistic
state, including a distinct final state when follow-up changes the counts.

```python
import numpy as np

from mdanderson_stats import U2OETCriteria, u2oet_scenario
from mdanderson_stats.u2oet_simulation import (
    U2OETAdaptiveSettings,
    simulate_u2oet_trial,
)
from mdanderson_stats.u2oet_fit import u2oet_parameter_names

names = u2oet_parameter_names(2, 2)
prior_mean = np.zeros(len(names) - 1)
prior_mean[[1, 2, 7, 8]] = 0.5
prior_sd = np.full(prior_mean.size, 0.2)
scenario = u2oet_scenario(
    np.broadcast_to([0, 1], (2, 2, 2)),
    np.broadcast_to([1, 0], (2, 2, 2)),
)
trial = simulate_u2oet_trial(
    [1, 2], [1, 2], scenario, [[10, 0], [100, 40]],
    prior_mean=prior_mean, prior_sd=prior_sd, initial=(0, 0),
    criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
    max_patients=4, cohort_size=2,
    draws=32, warmup=16, chains=2,
    adaptive_precision=U2OETAdaptiveSettings(
        target_mcse_ratio=0.05,
        initial_draws=32,
        max_draws_per_chain=64,
        batch_draws=32,
        max_total_work=100_000_000,
    ),
    rng=np.random.default_rng(20260929),
)
print(trial.final_precision_target_met, trial.final_precision_draws_per_chain)
print(trial.final_corner_mcse_ratio)
```

`U2OETAdaptiveSettings` requires the target, initial retained draws and maximum
retained draws per chain. Its whole-trial `max_total_work` is a caller-lowerable
cap on the worst-case number of likelihood-cell evaluations, conservatively
assuming a new fit at each of the maximum patient arrivals and a final fit.
Requests that exceed the cap or adaptive sampler's per-fit memory/work limits
are rejected before the caller's generator is advanced. The existing `draws`
argument is retained for fixed-mode compatibility and metadata; when adaptive
settings are supplied, `initial_draws` and the adaptive cap govern retained
sampling. `warmup` and `chains` remain common fit settings, and initial retained
draws must be at least warmup.

The trial raises `ArithmeticError` and makes no assignment if an adaptive fit
reaches its draw cap without meeting its target. It does not silently use an
under-precise posterior. `design_json` includes the adaptive settings so trial
aggregation can distinguish them. Each decision and the final result retain
only the target status, actual draws per chain, and the small chain-by-four
corner ratio matrix; posterior draw arrays are released after each fit. A
cached posterior and its diagnostics are reused when complete and
toxicity-only sufficient statistics are unchanged.

When adaptive mode is absent, fixed-fit calls, JSON metadata, and random-number
consumption remain on the existing path. Adaptive settings affect only the
posterior stream; separate data-stream seeds and calendar/outcome generation
are unchanged. The standalone adaptive wrapper covers PDS, CMI and PDS+CMI;
the GAO calendar uses a separate sampler and is not changed here.

The monitored MCSE/SD ratios concern only the four corner expected utilities.
Meeting the target does not certify precision for every parameter, interior
dose-pair utility, efficacy/toxicity risk, or downstream probability. The
trial's existing split-Rhat remains a separate diagnostic.
