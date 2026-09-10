# Original toxicity probability interval design

`TPIDesign` implements the original TPI method of Ji, Li and Bekele (2007),
*Clinical Trials* 4:235–244,
[doi:10.1177/1740774507079442](https://doi.org/10.1177/1740774507079442).
It extends coverage of MD Anderson catalog entry 72 alongside [mTPI](mtpi.md).
[Source provenance](tpi-sources.json) identifies the original paper and an author
presentation. Original software files are not redistributed.

## Posterior decisions

At each dose, the default prior is Beta(.005,.005); `prior=(a,b)` allows a common
alternative prior for sensitivity analysis. With y toxicities among n evaluable
patients, the posterior is Beta(a+y,b+n-y). Its standard deviation s sets interval
boundaries `target - lower_sd*s` and `target + upper_sd*s`. Intersecting these
intervals with [0,1] accommodates boundaries outside the probability domain.

The largest posterior interval mass chooses escalation, staying, or de-escalation.
Unlike mTPI, TPI does not divide masses by widths. Beta upper tails are computed
directly, and central mass uses the less cancellation-prone tail difference.
Numerical ties prefer de-escalation, then staying, then escalation.

Defaults `lower_sd=1.5, upper_sd=1` reproduce all columns of original Table 1.
The paper's partition prose puts K1 on the lower side, whereas its formal
probabilities put K2 there. The formal equations and numerical table correspond
to K1=1 on the upper side and K2=1.5 on the lower side. The author slides also
swap these labels between sections. The API uses explicit side names to avoid
this ambiguity. Custom multipliers require evaluation for the intended scenarios;
the defaults are not universally optimal.

## Safety and selection

A dose is unsafe if at least two patients have been evaluated and posterior
`P(p > target)` strictly exceeds `elimination_probability` (default .95).
The two-patient gate handles single-patient cohorts as described in the paper.
Unsafe doses are permanently excluded; carry the returned `eliminated` mask into
subsequent decisions. An unsafe lowest dose stops the trial. An unsafe current
dose requires de-escalation. Escalation into an excluded dose is forbidden, and
the remaining stay/de-escalation probabilities are compared.

At dose boundaries an outward move stays at the boundary dose; this explicit
Python convention avoids selecting an opposite-direction move from residual
tail probabilities. If the required adjacent dose is excluded, the trial stops
for safety. These edge conventions are not a native implementation audit.
Only doses identified as unsafe are individually excluded; lower-dose safety
stopping excludes every dose. Untried doses cannot be unsafe under the supplied
prior, which is checked when constructing the design.

Final selection uses isotonic posterior means of all tried doses, then chooses
the admissible dose closest to target. Equal-distance ties prefer the highest
at/below target, otherwise the lowest. Equal isotonic weights are the default;
explicit positive `weights` are accepted. Native weights and ambiguous ties
between distinct means on opposite sides of target remain unaudited.

## Usage

```python
import numpy as np
from mdanderson_stats import TPIDesign, simulate_tpi

design = TPIDesign()
table = design.decision_table(12)
assert list(table.action[:7, 5]) == ["E", "S", "S", "D", "DU", "DU", "DU"]
posterior = design.posterior(6, 1)
assert posterior.move == 0
np.testing.assert_allclose(posterior.probability.sum(), 1)

step = design.next_dose([3, 3, 0], [0, 3, 0], current_dose=2)
assert step.next_dose == 1
assert step.eliminated[1]
step = design.next_dose([6, 3, 0], [0, 3, 0], 1, eliminated=step.eliminated)
assert step.next_dose == 1

simulation = simulate_tpi(
    design,
    [0.1, 0.2, 0.3, 0.4, 0.5],
    cohorts=10,
    cohort_size=3,
    trials=5000,
    rng=72,
)
np.testing.assert_allclose(simulation.selection_probability.sum(), 1)
print(simulation.selection_probability)
```

`posterior` broadcasts count arrays and returns means, standard deviations,
interval boundaries, E/S/D probabilities, moves and safety summaries.
`decision_table` returns the shared `MTPITable` format: DLT rows, enrollment
columns, E/S/D actions with U marking unsafe doses. U takes precedence in trial
conduct even for custom priors/multipliers that produce a non-D raw action.
`select_mtd` returns `MTPISelection`, and `simulate_tpi` returns `MTPISimulation`;
these shared result containers do not imply use of mTPI decision rules.

Simulation precomputes all posterior decisions and batches binomial cohort draws
and dose updates with NumPy. It retains per-trial allocation, toxicity, exclusion,
selection and safety-stop results, with selection probabilities and Monte Carlo
standard errors. Selection bins are no MTD followed by one-based dose indices.
Complete outcomes and at most 200 patients, 100 doses, 100,000 trials and 2 million
trial-dose cells are supported. Priors have positive shapes at least 1e-6 and sum
at most 1e6; multipliers are between 1e-6 and 100.

Four focused checks cover the published table, independent weighted quadrature
for beta tails, the safety gate and exclusion/selection rules, and exhaustive
small-trial paths against batched simulation. These establish the stated Python
behavior, not native random-stream or full software parity.

**Entry 72 remains partial.** Scenario-based tuning, dose-specific informative
priors, posterior isotonic interval simulation, native archive audit and
spreadsheet/report workflows remain pending.
