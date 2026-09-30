# BOIN waterfall trials and simulation

`run_boin_waterfall_trial` and `simulate_boin_waterfall` provide complete-outcome
waterfall trials for a two-drug dose grid, with optional source-style single-
patient titration before the first subtrial. They complement the
[interactive next-subtrial planner](boin-combination.md), which operates on
accumulated data. A trial executes the initial staircase, subsequent row
searches and the special same-row search used by the original BOIN R simulator.

## Deterministic replay

The replay receives a toxicity-probability grid, one cohort budget per row and
a uniform-outcome tape. With titration disabled, it consumes one tape value
per assigned patient, in cohort order. A DLT occurs when that value is below
the assigned cell's toxicity probability. The tape has length
`sum(cohort_budgets)*cohort_size`; a stopped trial can leave the end unused.
Budgets apply in executed-subtrial order, not to fixed dose rows. Titration
uses the separate prefix and length described below.

```python
import numpy as np
from mdanderson_stats import BOINCombDesign, run_boin_waterfall_trial

trial = run_boin_waterfall_trial(
    BOINCombDesign(target=0.3),
    np.full((2, 3), 0.3),
    cohort_budgets=[4, 2],
    outcome_uniforms=np.full(18, 0.9),  # No DLTs in this replay.
)
assert trial.total_patients == 18
assert trial.total_toxicities == 0
np.testing.assert_array_equal(trial.patients, [[3, 0, 6], [3, 3, 3]])
assert trial.selected_contour == ((1, 3), (2, 3))
assert len(trial.subtrials) == 2
```

## Optional first-subtrial titration

With `titration=True` and `cohort_size > 1`, the first subtrial starts at the
first position of the initial staircase: down the first column, then across
the last row. The native waterfall wrapper starts here regardless of its
starting-dose value, so Python retains that convention. It assigns one patient
to each visited cell through the first DLT, or to the full staircase if no DLT
occurs. It then tops up the terminal cell by `cohort_size - 1` before applying
the usual first-subtrial movement rules. The top-up counts as the first cohort
of that subtrial; later cohorts use the full requested size. Cohort size one
disables titration, matching the native routine.

The replay reserves the first `rows + columns - 1` tape values for the full
initial staircase draw. Values after the first DLT are consumed but do not
represent assigned patients, matching the native vector draw. Remaining values
are consumed for top-up and ordinary cohorts. The required tape length is
`sum(cohort_budgets) * cohort_size + rows + columns - 2`; the simulation uses
the same fixed per-trial tape length. This is a Python replay convention for
deterministic assignment, not native RNG-stream parity. With `titration=False`,
the prior tape length and random draw sequence are unchanged.

```python
import numpy as np
from mdanderson_stats import BOINCombDesign, run_boin_waterfall_trial

trial = run_boin_waterfall_trial(
    BOINCombDesign(target=0.3),
    np.zeros((2, 3)),
    cohort_budgets=[1, 1],
    outcome_uniforms=np.full(9, 0.9),  # 4 staircase + 5 cohort slots.
    cohort_size=3,
    titration=True,
)
assert trial.titration_endpoint == (2, 3)
assert trial.titration_end_reason == "upper_dose_no_dlt"
np.testing.assert_array_equal(trial.titration_patients, [[1, 0, 0], [1, 1, 1]])
assert trial.total_patients == trial.patients.sum() == 6
```

`titration_patients` records only assigned staircase patients. The cohort trace
labels the single-patient run-in as `phase="titration"`, the initial top-up as
`phase="titration_topup"`, and ordinary enrollments as `phase="cohort"`.
The cohort number is phase-local: titration steps and the first cohort/top-up
can both be numbered one; `phase` distinguishes them. The first subtrial's
`start_position` remains one, the first actually assigned staircase position,
while `titration_endpoint` separately records where the ordinary first cohort
continues.
The result also reports the endpoint and whether titration ended at the first
DLT or at the upper dose without a DLT. Global stopping compares the actual
retained enrollment with the requested planned total; it does not infer
enrollment from the number of tape values consumed. The source checks the
planned total after completing a started subtrial, so actual enrollment can
exceed the requested cohort total by at most the staircase length minus one.
The Python preflight includes this overhead and keeps the maximum possible
actual enrollment at 1,000 patients.

Dose pairs, subtrial labels and starting positions are one-based.
`start_dose` identifies a position along the first staircase, rather than a
flattened grid index. The staircase descends the first column and then crosses
the final row. Later row searches contain columns 2 onward. The default
precision threshold is 12 patients and is checked at the destination dose
**after** a movement decision. `early_stop_patients` on this workflow controls
that threshold independently of the ordinary combination design's threshold.

The result retains actual patient/DLT matrices, exclusions, cohort assignments,
each subtrial's space/start/budget/candidate, final contour estimates and stopping
reasons. Counts include patients in a subtrial that fails to select a dose.
`row_candidates` precedes continuity, and `source_contour` retains the result
of the source continuity rule. `selected_contour` withholds a recommendation
if that destination is unobserved or excluded, recording it in
`continuity_blocked`. This conservative admissibility policy is an explicit
difference from native extrapolation; the original candidates remain visible.
Here `source_contour` means the source selection rule applied to the retained
Python observations, not a reproduction of R output after its dropped-count
bug.

## Serial simulation

```python
import numpy as np
from mdanderson_stats import BOINCombDesign, simulate_boin_waterfall

simulation = simulate_boin_waterfall(
    BOINCombDesign(target=0.3),
    [[0.05, 0.15, 0.3], [0.15, 0.3, 0.5]],
    cohort_budgets=[6, 4],
    trials=24,
    rng=981,
)
np.testing.assert_array_equal(simulation.total_patients, simulation.patients.sum(axis=(1, 2)))
np.testing.assert_array_equal(simulation.total_toxicities, simulation.toxicities.sum(axis=(1, 2)))
assert np.all(simulation.total_toxicities <= simulation.total_patients)
assert np.all(simulation.total_patients <= 30)
print(simulation.selection_probability)
```

Each trial gets an independent uniform tape, generated and consumed serially.
The integer seed is reproducible; an existing NumPy generator is advanced.
Results retain per-trial matrices and compact histories, plus mean allocation,
mean toxicity counts, row-specific selection probabilities and Monte Carlo
standard errors. Probabilities use the actual number of trials. Each row can
have no recommendation; this is not the same as all rows selecting a dose.
`selected_columns` contains one-based column labels, with zero for no selection.
`selection_probability` has one column per actual dose column, so a row sum
can be less than one. These probabilities are not conditioned on making a
recommendation. Compact histories preserve the source-continuity diagnostics.
`no_selection_probability` and `no_selection_mcse` report the omitted bin for
each row directly.

Resource limits are explicit: grids have 2 to 20 rows and columns with rows no
greater than columns; planned enrollment plus possible titration overhead is
at most 1,000 patients. A simulation has at most 100,000 trials, two million
combined result-array cells, 200,000 planned cohort and staircase records and
twenty million outcome draws. Invalid settings and output limits are checked
before advancing the supplied generator. Tapes and fits are constructed one
trial at a time.

## Statistical conventions and source differences

The reference is the full waterfall path in BOIN 2.7.2 `get.oc.comb`, not the
standalone rounded combination selector. Subtrial choices use weak-prior
posterior means, inverse-variance weighted one-dimensional isotonic regression
and a small ordered tie adjustment. The `bound_mtd` option applies the original
strict de-escalation bound to subtrial candidates. Final contour estimation
uses unrounded bivariate isotonic regression with weights `n+0.1`, a working
value of 1.1 at excluded cells, and the source row-continuity rule.
The returned `isotonic_estimate` precedes the final tie adjustment. Because
excluded cells enter the fit with a 1.1 working value, this matrix is not a
table of posterior probabilities; use the separate exclusion mask when
interpreting it.

Uniform Beta(1,1) tails govern ordinary safety and the final subtrial safety
check. The source's interim extra-safe check instead uses
`Beta(y+0.05,n-y+0.1)`; that distinction is retained explicitly. Caller-specified
indifference probabilities and safety offsets are honored, whereas the native
wrapper omits some of these arguments when constructing its boundary table.
Python applies the interim extra-safe rule once three patients are observed,
even when no ordinary elimination count exists at that sample size. Native R
nests that check inside availability of the ordinary boundary and can skip it.
Python also retains exclusions found by final subtrial safety; the native final
selector returns only its candidate and escalation flag.

The source wrapper can discard newly enrolled patients when a subtrial fails,
and some native percentage summaries divide by a fixed 1,000. Python preserves
all actual observations and uses the requested trial count. Native continuity
can also recommend an untreated cell; candidate and admissibility diagnostics
make the Python handling explicit rather than silently equating the outputs.

Validation records are in the [workflow audit](../research/boin-waterfall-workflow-audit.md)
and [combination source record](boin-combination-source.md). Deterministic
original-R references cover both local subtrial conduct and full transitions.
The full-wrapper reference replaces only the unavailable `Iso` call with an
independent exhaustive six-cell isotonic fit; existing original `Iso::biviso`
fixtures separately validate the shared numerical primitive.

The app's separate 3+3 run-in, integrated reports/protocols and desktop
executable parity remain outside this workflow.
