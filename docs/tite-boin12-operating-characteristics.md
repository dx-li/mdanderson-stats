# TITE-BOIN12 operating-characteristic simulation

`simulate_tite_boin12` runs independent TITE-BOIN12 calendar trials serially and
returns per-trial outcomes plus operating-characteristic summaries. Supply one
joint binary truth row per dose in the cell order
`(no toxicity/efficacy, no toxicity/no efficacy, toxicity/efficacy,
toxicity/no efficacy)`:

```python
import numpy as np

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12_simulation import simulate_tite_boin12

result = simulate_tite_boin12(
    BOIN12Design(0.35, 0.25),
    [[0.30, 0.35, 0.10, 0.25], [0.20, 0.25, 0.20, 0.35]],
    accrual_rate=0.2,
    toxicity_window=45,
    efficacy_window=60,
    cohorts=6,
    cohort_size=3,
    trials=100,
    arrival="exponential",
    seed=2026,
)
```

`arrival="fixed"` instead uses a gap of `1/accrual_rate` before every patient,
including the first. Conditional event delays are independent uniforms over
their endpoint windows; no-event endpoints remain unobserved until window
completion. These are explicit Python simulation choices, not native defaults.
Each trial receives separate arrival, outcome/timing, and sampler streams. The
returned `trial_seed_triplets` can be supplied in a later call instead of
`seed` to replay those same trial streams exactly.

Use `method="al"` for the existing TITE-BOIN12 AL conduct or `method="bda"`
with explicit `prior_concentrations`, `draws`, `warmup`, and `chains` for BDA.
The BDA prior may be shared across doses or specified dose by dose. Optional
conduct controls include pending-endpoint suspension thresholds and the
toxicity-limit-0.25 three-plus-three run-in.

`selected_obd` stores dose numbers, with zero meaning no selection. The result
also contains stop reasons, trial-level patient/toxicity/efficacy counts by
dose, accrual-stop and final ascertainment times, selection probabilities,
early-stop probability, and means with trial-level MCSEs. For one simulated
trial, MCSEs are undefined and returned as NaN. Selection MCSEs use the binomial
formula; other MCSEs treat trials as independent units. The function retains no
patient histories or BDA posterior draws across trials.

The runner preflights a conservative work estimate and bounded retained and
single-trial scratch sizes before creating random streams. These value limits
bound arrays, not total process RSS. See
[`research/tite-boin12-operating-characteristics-audit.md`](../research/tite-boin12-operating-characteristics-audit.md)
for source-backed assumptions and the limits of native-parity claims.
