# Keyboard dose finding

Catalog entry **127**, [Keyboard](https://biostatistics.mdanderson.org/shinyapps/Keyboard/),
has its single-agent statistical core implemented: posterior keys, dose decisions,
overdose safety, integer tables, isotonic MTD selection and batched trial simulation.
The entry remains **partial** for integrated protocol/report output. [Combination](keyboard-combination.md) and [time-to-event](tite-keyboard.md)
Keyboard implementations are documented separately.

Sources include the app's technical PDFs, the authors'
[statistical-properties paper](https://arxiv.org/abs/1712.06718), and independent
reference outputs from R **Keyboard 0.1.3**. The app snapshot identifies version
1.2.4.0, updated December 15, 2025. [Source hashes](keyboard-sources.json) pin the
retrieved evidence. Python implements the equations independently; R is used only
by the optional fixture-generation script.

```python
from mdanderson_stats import KeyboardDesign, simulate_keyboard

design = KeyboardDesign(target=0.3)
keys = design.posterior_keys(patients=6, toxicities=1)
print(keys.intervals, keys.probability, keys.strongest_key)
assert int(keys.move) == 1

step = design.next_dose([3, 6, 0], [0, 1, 0], current_dose=2)
assert step.next_dose == 3
mtd = design.select_mtd([3, 3, 15, 9, 0], [0, 0, 4, 4, 0])
assert mtd.dose == 3

simulation = simulate_keyboard(design, [0.05, 0.15, 0.3, 0.45, 0.6], trials=1000, rng=127)
print(simulation.selection_probability)  # no MTD, then doses 1 through 5
```

## Posterior keys

The target interval defaults to `(target-.05, target+.05)`; customize it with
`lower` and `upper`. Require target in [.05,.6] and
`0 <= lower < target < upper <= 1`. Extend this interval's width in both directions.
Under the uniform prior, dose counts `(n,y)` yield Beta(y+1,n-y+1).

The strongest key determines the ordinary decision: below the target key means
escalate, above means deescalate, and the target key means stay. Public key and dose
indices are one-based. `posterior_keys` broadcasts counts and returns actual key
probabilities, decision scores and moves (+1, 0, -1). Numerically tied strongest
scores select the highest key, using a relative tolerance of 32 machine epsilons.

Two edge conventions are explicit because the published paper and R differ:

* `edge_rule="discard"` (default) retains only full-width keys, as in the paper.
  For target .2 and interval (.15,.25), the keys cover (.05,.95). Their probability
  masses need not sum to one; discarded endpoint mass is not redistributed.
* `edge_rule="rescale"` includes shortened endpoint keys and scores them as
  `probability * full_width / actual_width`, matching R 0.1.3's convention.
  The returned `probability` still contains true posterior mass, distinct from
  the width-adjusted `score`.

Lower and upper beta tails are evaluated directly to avoid subtracting two values
near one. Counts are limited to the app's 200-patient scope; key width must be at
least .001, bounding computation to about 1,000 intervals. Endpoints within
floating-point rounding tolerance of zero or one are snapped to those endpoints.
No-data dose decisions are rejected. Untreated doses are valid in trial snapshots
and final MTD estimation.

## Conduct and safety

`next_dose` accepts matching patient and DLT count vectors for 2..100 doses, with
at most 200 patients in total. Dose transitions are by one level, clipped at the
range boundaries; eliminated doses cannot be revisited. Pass the returned
`eliminated` mask into subsequent decisions and selection to retain prior exclusions.

Eliminate a dose and all higher doses if at least three patients have been treated
there and its Beta(1,1)-prior posterior `Pr(p>target)` is strictly greater than
`elimination_probability` (default .95). With `extra_safe=True`, the lowest dose
also triggers stopping at cutoff `elimination_probability-safety_offset` (.90 by
default), again after at least three patients. Lowest-dose elimination stops with
no MTD. Safety overrides ordinary dose decisions and precision stopping.

`early_stop_patients=m` stops once m patients have been treated at the current dose,
regardless of the ordinary next move, provided safety does not require deescalation
or stopping. This is different from BOIN's stay-dependent precision stop. The option
is off by default. `boundary_table` supplies inclusive DLT cutoffs for every sample
size; escalation cutoff -1 means never escalate, and safety/deescalation cutoff
`n+1` means impossible. Its last column is the lowest-dose stopping threshold.
The table and decision use the same rules, including custom target intervals.

## MTD selection and simulation

MTD selection shares the weak Beta(.05,.05) prior and inverse-variance weighted
isotonic mean estimator used by BOIN. Fit only treated, noneliminated doses for
selection and choose the mean closest to target. Ties choose the highest dose if
all tied means lie below target, otherwise the lowest. No admissible dose gives
`dose=None`. Reporting means include all treated doses. Untreated estimates are NaN.

Keyboard's `marginal_interval` reports each dose's individual beta .025/.975
quantiles, without isotonic pooling of intervals. `report_overdose_probability`
uses weak-prior posterior probabilities with unweighted isotonic pooling, whereas
`safety_overdose_probability` uses the uniform safety prior without pooling.
The reporting interval is not a joint posterior band.

`simulate_keyboard` assumes independent Bernoulli DLT outcomes, all evaluated
before the next cohort. It precomputes the decision table and advances cohorts
across trials with NumPy arrays. Defaults are 10 cohorts of 3, starting at dose 1,
with 1,000 trials. Returned trial-level patient/toxicity arrays have shape
`(trials,doses)`. Selection index zero means no MTD. Probability and Monte Carlo
standard-error bins are `[no MTD, dose 1, ..., dose J]`. Stop reasons distinguish
safety, precision and maximum enrollment. NumPy random draws differ from R seeds.

## Reference discrepancies and validation

R 0.1.3's boundary generator can print elimination below three patients, while its
conduct and selection guard against that. Python consistently uses the three-patient
minimum. The older app Guide says greater than three for extra safety, but the
native conduct rule uses at least three. R's simulator also fails to forward its
custom margins and cutoff into boundary generation and some selection settings;
Python honors the supplied design throughout. Native simulation comparisons use
default settings. Safety takes precedence over precision stopping in Python, whereas
R checks its precision threshold first. R perturbs fitted means slightly to break
MTD ties; Python resolves ties without changing estimates.

Validation compares six native full decision tables, 64 original MTD selections
and reported means, and a 10,000-trial native simulation. Independent exact rational
binomial-tail identities check beta key probabilities, while deterministic paths
check custom-key forwarding and safety/precision precedence. Native elimination
comparisons start at three patients because of the documented table inconsistency.
