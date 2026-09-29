# MTADF logistic trial simulation

`simulate_mtadf_logistic` runs complete-cohort trials using the
[global or local logistic design](mtadf-logistic.md). It generates independent
binary toxicity and efficacy outcomes at the assigned dose, applies the
existing safety and efficacy rules, and reports allocation, final selection
and stopping. The independence of generated outcomes is an explicit simulation
assumption; the marginal decision models do not require it.

```python
import numpy as np
from mdanderson_stats import (
    MTADFLogisticSimulationConfig,
    simulate_mtadf_logistic,
    simulate_mtadf_logistic_trial,
)

config = MTADFLogisticSimulationConfig(
    model="global",
    true_toxicity=[0.05, 0.10, 0.30],
    true_efficacy=[0.15, 0.50, 0.40],
    doses=[-1.0, 0.0, 1.0],
    cohorts=4,
    cohort_size=3,
    draws=256,
    warmup=256,
    chains=2,
)
simulation = simulate_mtadf_logistic(config, trials=3, rng=940)
replay = simulate_mtadf_logistic_trial(
    config,
    outcome_seed=int(simulation.trial_seeds[0, 0]),
    sampler_seed=int(simulation.trial_seeds[0, 1]),
)
np.testing.assert_array_equal(replay.subjects, simulation.subjects[0])
assert replay.selected_dose == simulation.selected_dose[0]
print(simulation.selection_probability, simulation.no_selection_probability)
```

The deliberately small example demonstrates the API and reproducibility. Its
trial count and short posterior chains do not establish operating
characteristics or adequate posterior convergence. Choose dose coding, chain
lengths and study replication deliberately for an actual analysis.

For local logistic trials, change the model and specify the local window and
slope thresholds as needed:

```python
from dataclasses import replace

local_config = replace(config, model="local", window_length=2)
local_simulation = simulate_mtadf_logistic(local_config, trials=3, rng=941)
print(local_simulation.selection_probability)
```

## Conduct and final selection

Trials start at the lowest admissible dose. The local design first visits one
cohort at each of its first `window_length` levels, subject to safety, and then
uses the local slope and previously treated next-dose rules. Safety stops and
unsafe-current de-escalation take priority over efficacy fitting. No posterior
is needed for an action determined solely by safety or the initial local ramp.

At maximum enrollment, global trials select the admissible dose with highest
posterior mean efficacy, potentially including an untried dose. Local trials
use final double-sided isotonic selection from observed doses. No admissible
candidate produces no selection. `selected_dose` is zero-based, with `-1`
denoting no selection. Early stopping means enrollment was below the planned
sample size; a final safety stop is no selection but is not early stopping.

The output retains trial-by-dose subject, toxicity and response counts,
per-trial reasons and compact sampler diagnostics. Selection and no-selection
probabilities use all simulated trials as their denominator and include
Monte Carlo standard errors. Their probabilities sum to one. Mean dose-level
counts describe allocation and observed outcomes.

## Reproducibility and resource use

Every trial receives separate outcome and posterior-sampling seeds. The
returned `trial_seeds` permits replay through `simulate_mtadf_logistic_trial`.
Changing sampler effort does not directly consume the outcome random stream;
it can still change decisions and therefore which doses receive those outcomes.

Trials and posterior chains run serially. Posterior draws are released between
reviews; the aggregate retains compact summaries rather than patient-by-draw
or trial-by-draw arrays. Work bounds include final global fits and possible
local next-window fits. Shared boundary windows reuse one posterior. Requests
over the configured work or storage budget raise before simulation.

This implements the published logistic methods with the package's documented
complete-cohort conduct and sampling choices. It does not claim native
application random-stream, calendar-time or report parity. See the
[implementation audit](../research/mtadf-logistic-simulation-audit.md).
