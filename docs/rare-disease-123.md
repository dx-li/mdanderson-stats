# 1+2+3 dose finding for rare diseases

`RareDisease123Design` implements the complete-outcome dose-assignment rules
from the [MD Anderson 1+2+3 app](https://biostatistics.mdanderson.org/shinyapps/1plus2plus3/),
catalog entry 172. The app identifies authors Shuqi Wang, Bingming Yi, Josh Chen
and Ying Yuan, version V1.0.0.0, PID 1188, updated April 6, 2026. Its reference
is titled *1+2+3 Design for Rare Diseases with Application To Gene Therapy* (2026).
The implementation was based on its downloaded trial protocol, help PDFs,
generated flowchart and decision tables. [Provenance](rare-disease-123-sources.json)
records these artifacts. Original documents and application files are not redistributed.

## Inputs and verified scope

Supply an explicit `efficacy_prior=(a,b)` for independent beta-binomial efficacy
updating. **The public protocol does not state this prior.** Beta(1,1) reproduces
all 69 cells of the captured default tables for 1, 3 and 6 patients, but this does
not identify the native prior for every setting. Requiring the prior makes this
uncertainty visible instead of supplying an undocumented default.

Defaults otherwise follow the live app: toxicity target .28, minimum efficacy .6,
efficacy admissibility cutoff .90, and escalation cutoff .50. The efficacy help
PDF recommends .95 whereas the live app defaults to .90. Both are configurable.
Escalation cutoff must exceed one minus the efficacy cutoff, as required by the
help document. Toxicity targets .05–.60 and efficacy targets .10–.90 are supported.

The toxicity boundaries reuse `BOINDesign` with its standard alternative rates.
At target .28 they round to .221 and .334, matching the generated protocol.
Calculations use unrounded boundaries. Efficacy posterior probability is evaluated
using beta CDF/upper-tail functions without posterior sampling.

## Cohorts, admissibility and decisions

Begin by treating one patient at the prespecified starting dose. Pass complete
per-dose counts to `next_dose`: patients, DLTs, responses, and the current one-based
dose index. DLT and response counts can overlap within patients; only their
marginal counts enter these decisions. Dose vectors contain 2–20 doses and patient
counts must be 0, 1, 3 or 6. Reassigning a dose with one patient adds two; reassigning
one with three adds three. The maximum per-dose enrollment is six.

Untried and one-patient doses are admissible unless already eliminated. For doses
with at least three patients, admissibility requires observed toxicity strictly
below the BOIN de-escalation boundary and posterior `P(efficacy < minimum)`
strictly below the efficacy cutoff. A toxicity exclusion removes that dose and
all higher doses; a futility exclusion removes that dose only. Carry the returned
`eliminated` boolean mask into subsequent decisions to retain exclusions.

The decision rules follow the downloaded protocol:

- In the high-toxicity region (at or above the de-escalation boundary), move down
  if the adjacent lower dose is admissible, otherwise stay if the current dose
  is admissible, otherwise stop with no OBD.
- In the intermediate region, stay at an admissible current dose; otherwise
  escalate if the adjacent higher dose is admissible and has at most one patient;
  otherwise stop with no OBD.
- In the low-toxicity region (at or below the escalation boundary), escalate when
  posterior `P(efficacy >= minimum)` is below the escalation cutoff and the higher
  dose is admissible with at most one patient. Otherwise stay if the current dose
  is admissible; otherwise stop with no OBD.

A nonexistent neighbor is unavailable. These are adjacent-dose rules, with no
skipping over excluded doses. After deciding the next assignment, if that dose
already has six patients, select it as OBD and terminate. Filling the current dose
to six does not alone terminate: the rule can send the next cohort elsewhere.
The exploration restriction follows the protocol's explicit `n[j+1] <= 1` and
its decision-table footnotes; the flowchart describes this more loosely as higher
dose admissibility.

## Usage

```python
from mdanderson_stats import RareDisease123Design

design = RareDisease123Design(efficacy_prior=(1, 1))
step = design.next_dose([1, 0, 0], [0, 0, 0], [1, 0, 0], current_dose=1)
assert step.action == "stay" and step.cohort_size == 2
step = design.next_dose([3, 0, 0], [0, 0, 0], [3, 0, 0], 1, eliminated=step.eliminated)
assert step.cohort_size == 3
step = design.next_dose([6, 0, 0], [0, 0, 0], [6, 0, 0], 1, eliminated=step.eliminated)
assert step.action == "select_obd" and step.selected_dose == 1
assert step.next_dose is None
```

`RareDisease123Decision` returns the action, next dose and cohort size, selected
OBD if terminal, current admissibility, persistent exclusions and posterior
efficacy upper-tail probabilities. Terminal actions have no next dose and cohort
size zero. Array summaries are read-only.

Three focused tests check every default native table cell, cohort progression
and selection at the next assigned full dose, the one-patient exemption,
toxicity/futility exclusions, the escalation exploration limit and stopping.

## Operating characteristics

`simulate_rare_disease_123` runs independent trials in batches, using the same
rules as `next_dose`. Outcomes are available after each complete cohort. The
app's simulation help specifies correlated standard normals, thresholded at
normal quantiles of the dose-specific toxicity and efficacy probabilities.
`correlation` therefore specifies **latent normal correlation**, not the Pearson
correlation of the observed binary endpoints. Its default is .1, matching the app.
Endpoint probabilities 0 and 1 and latent correlations -1 and 1 are supported.

```python
from mdanderson_stats import RareDisease123Design, simulate_rare_disease_123

simulation = simulate_rare_disease_123(
    RareDisease123Design(efficacy_prior=(1, 1)),
    toxicity_rates=[0.1, 0.25, 0.4],
    efficacy_rates=[0.7, 0.7, 0.7],
    trials=10000,
    rng=172,
)
assert abs(simulation.selection_probability.sum() - 1) < 1e-12
assert simulation.patients.max() <= 6
```

Selection probabilities and their Monte Carlo standard errors are indexed
`[no OBD, dose 1, ..., dose J]`. Per-trial `selected_dose` uses 0 for no OBD and
one-based dose indices otherwise. Patient, toxicity and response counts and
persistent exclusions are returned per trial and dose; their means are per dose.
Arrays are read-only. `start_dose` defaults to 1. At most three cohorts and six
patients can be enrolled per dose. Simulations support up to 100,000 trials.

Focused checks compare selection probabilities and allocation means with exhaustive
small-trial paths, cover deterministic endpoint extremes, and verify perfectly
correlated binary outcomes. Seed reproducibility is for this implementation;
matching the native application's random-number sequence is not claimed.

**Catalog status is partial.** Generalized 1+a+b cohorts, within-cohort staggering,
generated reports and native prior/source audit remain pending. This is not a
claim of complete app parity.
