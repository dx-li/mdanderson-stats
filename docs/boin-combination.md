# BOIN drug-combination designs

The Python implementation covers ordinary combination dose decisions and final
MTD selection, including an MTD contour, plus the waterfall planner used to move
between subtrials. It corresponds to catalog entry 128,
[BOINComb](https://biostatistics.mdanderson.org/shinyapps/BOINComb/).
The [source audit](boin-combination-source.md) records the original package,
app versions, numerical references and backend differences.

```python
import numpy as np
from mdanderson_stats import BOINCombDesign

design = BOINCombDesign(target=0.3)
patients = np.array([[3, 0, 0], [0, 0, 0]])
toxicities = np.zeros_like(patients)
decision = design.next_dose(patients, toxicities, (1, 1), rng=128)
assert decision.next_dose in ((1, 2), (2, 1))

selection = design.select_mtd(patients, toxicities)
assert selection.dose == (1, 1)
```

## Conduct and safety

Dose matrices have drug A levels in rows and drug B levels in columns, with
at least two levels of each drug and no more rows than columns. Dose pairs
use one-based labels. Counts represent fully evaluated binary DLT outcomes,
with at most 1000 total patients;
toxicity is assumed nondecreasing with either drug's dose.

The current dose's observed DLT count is compared with the ordinary BOIN
integer boundaries. Escalation and de-escalation change one drug by one level.
Eligible neighbors are ranked by posterior probability between the BOIN
escalation and de-escalation boundaries, using a Beta(.5,.5) prior and the
original `.0005 * patients` adjustment. Supply a NumPy generator or seed for
reproducible random ties.

Safety uses a separate Beta(1,1) prior. After at least three evaluated patients,
a posterior probability of exceeding the target above the elimination cutoff
excludes that combination and its southeast rectangle during conduct.
`extra_safe=True` tightens stopping at the lowest combination. Retain the
returned `eliminated` mask when making later decisions or selecting an MTD.
No eligible dose is represented by `None` rather than the R sentinel `(99,99)`.
`early_stop_patients` defaults to 100; `None` disables it. The interactive
`next_dose` helper stops at that current-dose count, following `next.comb()`.
`desirability_table` returns the posterior interval mass for the supplied grid,
with excluded cells set to negative infinity; the neighbor scoring rule also
adds the sample-count adjustment described above.

## Final selection

`select_mtd` fits weak-prior means `(toxicities+.05)/(patients+.1)` using
weighted bivariate isotonic regression, with weights `patients+.1`. Untreated
cells contribute to the fit but cannot be selected. Returned estimates retain
full precision. The native selection rule rounds the working estimates to two
decimals and adds a small row-plus-column adjustment before comparing them
with the target. Remaining ties follow column-major order.

The source's final safety rule uses row/column exclusions, distinct from the
rectangular conduct rule. Supplying the conduct mask preserves exclusions
already made during a trial. `bound_mtd=True` restricts final candidates by the
de-escalation boundary. `mtd_contour=True` returns the row-wise contour using
the native highest-column continuity rule. All-untreated and all-eliminated
data return no MTD.

## Waterfall subtrials

The initial search path goes down the first column, then across the last row.
Later subtrials search columns 2 through the last column of a lower drug A row.
`next_subtrial` is called after a subtrial is complete, with accumulated trial
counts. It identifies the latest occupied subtrial and uses weighted
one-dimensional isotonic selection to choose the next start and search space.
The planner does not itself enroll patients or decide when a subtrial ends.

```python
from mdanderson_stats import next_subtrial

patients = [[6, 0, 0, 0], [6, 10, 12, 0], [9, 12, 0, 0]]
toxicities = [[0, 0, 0, 0], [1, 1, 4, 0], [2, 3, 0, 0]]
plan = next_subtrial(0.3, patients, toxicities)
assert plan.starting_dose == (1, 4)
assert plan.next_subtrial == ((1, 2), (1, 3), (1, 4))
```

## Simulation

```python
from mdanderson_stats import simulate_boin_combination

simulation = simulate_boin_combination(
    BOINCombDesign(target=0.3),
    [[0.05, 0.10, 0.20], [0.15, 0.25, 0.40]],
    cohorts=8,
    cohort_size=3,
    trials=1000,
    rng=128,
)
print(simulation.selection_probability)
print(simulation.selection_mcse)
```

Each cohort has fully observed independent binomial DLT outcomes. Results
retain per-trial patient and toxicity matrices, exclusions, final selections
and stopping reasons. Selection probabilities and their binomial Monte Carlo
standard errors use one-based matrix bins, with `(0,0)` reserved for no MTD.
R and NumPy use different random streams; equal seed values do not imply equal
trial histories.

The simulator follows `get.oc.comb()`'s unrounded final selection, omits the
interactive helper's empirical escalation blockers, and stops at the enrollment
threshold only when its convergence condition is met. `next_dose` exposes this
movement variant as `source_simulation=True`; callers using that flag must apply
the convergence stop themselves. `select_mtd(round_selection=False)` exposes
the corresponding unrounded selection. Safety exclusions remain active.

A local 2,000-trial run of the example scenario took approximately 3.03 seconds
and enrolled 24 patients per trial. Compared with an independent 2,000-trial
BOIN 2.7.2 run, all six selection frequencies differed by less than 1.72 combined
Monte Carlo standard errors. This checks one scenario, not every design setting.
Independent bivariate-isotonic reference cases had maximum absolute error
`1.21e-8`. Focused checks also cover safety precedence and escalation beyond an
enrollment threshold when convergence has not yet occurred.

## Scope

Accelerated titration, moderate-toxicity stopping during titration, the app's
3+3 run-in, full waterfall simulation, enumerated desirability-rank tables and
generated trial protocols/reports remain outside this API.
These are tracked as outstanding coverage rather than inferred from the
ordinary combination movement rule.
