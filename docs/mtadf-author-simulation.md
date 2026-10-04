# MTADF author isotonic simulation

The repository's `simulate_mtadf` follows the paper-policy implementation
documented in [MTADF](mtadf.md). The rendered author R file also supplies a
distinct `isotonic()` operating-characteristic procedure. This module makes
that procedure replayable and simulatable while keeping its behavior
separate.

```python
import numpy as np

from mdanderson_stats.mtadf_author_simulation import (
    replay_mtadf_author_trial,
    simulate_mtadf_author,
)

# Rows are cohorts; columns are potential binomial counts by dose.
toxicity_tape = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
response_tape = np.array([[1, 2, 0], [2, 1, 0], [1, 1, 0]])
trial = replay_mtadf_author_trial(toxicity_tape, response_tape)
print(trial.assigned_dose, trial.selected_dose)

oc = simulate_mtadf_author(
    true_toxicity=[0.05, 0.20, 0.35],
    true_efficacy=[0.10, 0.50, 0.40],
    cohorts=4,
    cohort_size=3,
    trials=20,
    rng=274,
)
print(oc.selection_probability)
```

`replay_mtadf_author_trial` accepts complete potential cohort outcome tables
with shape `(cohorts, doses)`. It applies only the potential counts for the
assigned dose. The starting dose is zero-based dose 0. It returns the assigned
dose and the admissible-dose cap before and after each cohort, plus dose-level
counts and the final decision. Outcomes are aggregated binomial counts; the
simulation assumes independent toxicity and efficacy margins. The simulator
generates potential cohort counts for each dose and endpoint, then calls the
same replay. It uses NumPy's generator and does not reproduce R's random-number
stream.

The author procedure computes the current cohort's next-dose movement using
the admissibility cap cached before that cohort, and refreshes the cap after
choosing the next dose. The cap always retains at least the lowest dose, so
this procedure does not have an early safety stop. Final selection instead
computes a fresh cap and uses the author all-dose efficacy fit, with a
`0.0001` denominator offset and the rightmost fitted maximum. Consequently an
untried dose can be selected on a final efficacy tie. These are source
semantics, not recommendations for clinical use.

The source `Iso` package's unconstrained `ufit` behavior is defined for a
prefix of at least two doses. Its one-dose call at the first cohort has
undefined internal indices. This Python replay uses the one observed rate
unchanged for that singleton prefix so the trial can proceed; this edge is an
explicit completion policy, not a claim of R equivalence.

The rendered R routine's prior is fixed by solving
`pbeta(0.3, alpha, 0.5-alpha) = 0.22`; the public replay retains the source
toxicity-limit and posterior-cutoff defaults (`0.3` and `0.8`). Dose-level
limits are 20, cohorts and planned patients are each bounded at 1,000, and
serial simulation is bounded by decision and potential-outcome work limits.
The reported selection Monte Carlo standard errors use all requested trials.
Small runs demonstrate mechanics and seeded reproducibility, not precise
operating characteristics. See the
[source and numerical audit](../research/mtadf-author-simulation-audit.md).
