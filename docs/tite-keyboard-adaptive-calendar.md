# Adaptive TITE-Keyboard calendar trials

The adaptive timing posterior can be used at each interim decision in a calendar
replay and in serial operating-characteristic simulations. Timing priors are
explicit; this path does not claim native-app defaults or RNG parity. The
simulated event-time distribution is separate from the fitted timing model.

```python
import numpy as np

from mdanderson_stats import KeyboardDesign
from mdanderson_stats.tite_keyboard_adaptive_calendar import TITEKeyboardAdaptiveSettings
from mdanderson_stats.tite_keyboard_simulation import simulate_tite_keyboard

settings = TITEKeyboardAdaptiveSettings(
    lambda_prior=(0.5, 0.5),  # explicit Gamma(shape, rate)
    gamma_prior=(0.5, 0.5),
    chains=4,
    warmup=1_000,
    draws=4_000,
    max_fit_work=50_000_000,
    max_total_work=100_000_000,
)
result = simulate_tite_keyboard(
    KeyboardDesign(target=0.3),
    true_toxicity=[0.15, 0.30, 0.45],
    window=90.0,
    accrual_rate=0.3,
    cohorts=2,
    cohort_size=3,
    trials=2,
    event_distribution="uniform",
    pending_fraction_limit=None,
    rng=2718,  # explicit integer replay seed required in adaptive simulation
    adaptive_timing=settings,
)
print(result.adaptive_fit_count, result.adaptive_work_units)
print(result.max_adaptive_split_rhat, result.max_adaptive_weight_mcse)
print(result.outcome_seed, result.sampler_seed)
```

At each decision, only DLTs whose event time is at or before that calendar time
enter the observed-event likelihood. Pending follow-up ages are measured at the
same decision time. Future potential event times are not exposed to the timing
sampler. If a decision has no pending patients, the ordinary complete-data
controller is used without a timing fit. In adaptive mode, observed DLT event
ages must be strictly inside the assessment window; an endpoint event that
would enter an adaptive fit is rejected because the timing density is defined
on the open interval. Timing fits are also skipped when an immediate safety
stop, current-dose elimination, or pending-fraction suspension determines the
controller action independently of adaptive weights.

Each adaptive step retains compact parameter and weight diagnostics, not MCMC
draws. Failed diagnostic thresholds raise and stop the trial/simulation rather
than silently contributing an OC result. `max_fit_work` bounds any one fit and
`max_total_work` is enforced over all fits, including suspension reevaluations
and all simulated trials. The per-fit sampler preflights its conservative work
bound before consuming its Generator; aggregate work is checked before every
subsequent fit. Adaptive simulation limits retained trial-by-dose output to two
million cells.

Adaptive simulation requires an integer seed. It deterministically derives
separate outcome and sampler seeds and returns both, along with aggregate fit
work and maximum observed diagnostics. The timing stream generates one explicit
fit seed per trial in replay order. Single-trial `run_tite_keyboard_trial`
instead accepts an explicit `adaptive_rng` Generator. Ordinary calendar paths
retain their existing RNG behavior when `adaptive_timing` is omitted.
