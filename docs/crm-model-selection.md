# CRM model selection and Occam's window

`fit_bmacrm` supports Bayesian model averaging (`aggregation="bma"`, the default),
Bayesian model selection (`"bms"`), and averaging within Occam's window
(`"occam"`). These use the same power-model posterior integrations.

```python
from mdanderson_stats import fit_bmacrm, bmacrm_decision

settings = dict(
    skeletons=[[0.05, 0.15, 0.30, 0.50], [0.10, 0.25, 0.45, 0.65]],
    events=[0, 1, 1, 0],
    subjects=[3, 3, 3, 0],
    target=0.30,
)
selected = fit_bmacrm(**settings, aggregation="bms")
windowed = fit_bmacrm(**settings, aggregation="occam", occam_threshold=0.6)
print(selected.posterior_model_weights)
print(selected.aggregation_model_weights)
print(windowed.aggregation_model_weights)
print(bmacrm_decision(selected, current_dose=2).dose)
```

The [Yin and Yuan paper, Section 2.3](https://saasresearch.hku.hk/~gyin/materials/2009YinYuanJASA.pdf)
defines BMS by choosing the model with the largest posterior probability at
each decision. Occam's window keeps a model when its posterior probability
divided by the largest probability is strictly greater than a threshold.
The retained models are averaged with their renormalized probabilities. The
[online BMACRM app](https://biostatistics.mdanderson.org/shinyapps/BMACRM/)
exposes BMA and BMS; Occam's window is a separate paper-based option here.

`posterior_model_weights` always reports the full posterior probabilities.
`aggregation_model_weights` reports the weights actually used for `dose_mean`
and `overdose_probability`. Per-model evidence, means and parameter summaries
are unchanged by the aggregation choice. BMS uses a single model's overdose
probabilities; Occam uses the retained mixture for both estimation and safety.

BMS chooses the first input model when log posterior masses tie exactly.
Comparisons use log masses before normalization, so very small prior weights
can recover when the data support them. Occam requires an explicit finite
`occam_threshold` in `[0, 1)`. At zero, all models with positive prior mass are
eligible. A threshold equal to a model's weight ratio excludes that model;
no tolerance band changes this strict comparison. Thresholds are rejected for
the other aggregation modes.

## Trial conduct and simulation

`crm_calendar_decision`, `run_crm_trial` and `simulate_crm` accept the same
aggregation arguments for `method="bmacrm"`. DA-CRM accepts the default
aggregation only. `bmacrm_lookahead` inherits the fitted posterior's settings.
Each completed-outcome scenario recomputes model evidence and selection using
the original skeletons and prior weights. A model excluded at an earlier
decision can reenter; exclusion is never written back into its prior.

```python
from mdanderson_stats import simulate_crm

simulation = simulate_crm(
    [[0.05, 0.15, 0.30], [0.10, 0.25, 0.45]],
    true_toxicity=[0.05, 0.20, 0.40],
    window=1,
    accrual_rate=0.5,
    target=0.25,
    aggregation="bms",
    cohorts=2,
    cohort_size=1,
    trials=2,
    event_distribution="uniform",
    rng=6204,
)
print(simulation.selection_probability, simulation.no_selection_probability)
```

These trial helpers retain the documented Python CRM Suite conduct policy and
calendar scheduling. Combining it with BMS or Occam's window is an explicit
Python extension. It does not reproduce the original JASA trial algorithm:
that paper restricts both escalation and deescalation to one level, while the
Suite has its own no-skip, raw-rate and final-selection rules. The paper's
simulation tables also use alpha prior standard deviation 2; the package's
established default is `sqrt(2)`. Supply `prior_sd=2` when that prior is intended.

Work remains serial and uses the existing evaluation budgets. Selection adds
small vector operations and does not allocate a simulation-by-draw tensor.
Small examples demonstrate usage, not precise operating characteristics.

## Evidence and remaining scope

Independent base-R posterior references support the selected and windowed
estimates, including zero prior mass and a tiny prior rescued by data. Focused
checks cover selection boundaries, model reentry and propagation through trial
conduct. See the [implementation audit](../research/crm-model-selection-audit.md).

Catalog entry 133 has partial scientific coverage. The online application's
hidden prior and conduct conventions, automatic skeleton calibration, native
output and random-sequence equivalence remain unverified. This implementation
does not claim those behaviors. Native files and older desktop conduct
differences for entries 81 and 132 also remain open.
