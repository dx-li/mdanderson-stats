# BARD paper BF-BLRM decisions

These functions implement the paper's stage-one BF-BLRM rules using posterior
target (PTT) and overdose (POD) probabilities from the [model fit](bard-blrm.md).
They use one-based dose indices. The probabilities must describe the same
ordered doses, target interval and posterior; POD must be nondecreasing.
The helpers accept at most 100 doses and return immutable result arrays.

## Dose movement and boundaries

Among doses with `POD < eta`, `bard_blrm_next_dose` identifies the greatest
PTT and moves one level toward that target from `current_dose`. Exact PTT ties
select the lowest dose, an explicit Python convention. The paper recommends
`eta=0.30`, which is the helper default; it is not an inferred native app
setting. All POD values strictly above eta produce `stop_all_overdose`.
If equality leaves no safe dose, the separate `no_eligible_safe_dose` action
has no next dose. Equality is a real possibility with sampled probabilities.

```python
from mdanderson_stats import bard_blrm_next_dose

decision = bard_blrm_next_dose(
    ptt=[0.1, 0.3, 0.4, 0.2],
    pod=[0.01, 0.05, 0.1, 0.35],
    current_dose=1,
    eta=0.3,
)
assert decision.target_dose == 3
assert decision.next_dose == 2 and decision.next_dose_safe
```

For fitted values, pass `fit.posterior_target_probability` and
`fit.posterior_overdose_probability` from the same fit. Inspect Monte Carlo
errors and convergence diagnostics before using probabilities near a cutoff.

A downward step can remain unsafe when several doses separate the current
dose from the safe target. For example, POD `[.1, .35, .6]`, current dose 3
and target dose 1 imply a one-step move to dose 2 with `next_dose_safe=False`.
The helper exposes both the paper's movement rule and its safety result.
It does not authorize assignment at that unsafe intermediate dose; resolving
that scheduler case requires an explicit protocol policy. No automatic skip
or hidden assignment rule is supplied.

## Backfill and final MTD

`bard_blrm_backfill` opens a dose below the current escalation dose only when
a response has been observed there or at a lower dose. A dose closes when
its own POD is at least eta or its toxicity-evaluable patient count reaches
`cap`. The highest open dose is selected. `eligible` is the mask after both
eligibility and closure rules; `closed` records the POD/cap closure condition.

Observed responses, completed toxicity assessments and assigned patients are
separate counts. Supply `assigned` when any toxicity outcomes are pending;
otherwise it defaults to `evaluable`. A response can be observed before its
patient's toxicity assessment completes. The paper's cap counts evaluable
patients, so this helper does not itself limit outstanding assignments.

```python
from mdanderson_stats import bard_blrm_backfill, bard_blrm_select_mtd

backfill = bard_blrm_backfill(
    pod=[0.1, 0.2, 0.4],
    responses=[1, 0, 0],
    evaluable=[0, 0, 0],
    assigned=[2, 1, 0],
    current_dose=3,
    cap=4,
)
assert backfill.selected_dose == 2

selection = bard_blrm_select_mtd(ptt=[0.5, 0.4, 0.4], pod=[0.1, 0.2, 0.4], treated=[5, 6, 12])
assert selection.selected_dose == 2
```

Final MTD selection maximizes PTT among doses with `POD < eta` and at least
six treated patients. Include escalation and backfill patients in `treated`
and use a final fit incorporating their completed toxicity outcomes. The
minimum is configurable explicitly; six follows the paper. Exact ties again
favor the lowest dose. No qualifying dose returns `None` with a status that
distinguishes all-overdose, cutoff equality and insufficient treated counts.

Twelve independent hand-calculated snapshots cover movement, cutoff equality,
ties, response timing, cap closure and final eligibility. Four focused worker
checks and a separate mathematical review also pass; see the
[source and validation audit](../research/bard-blrm-audit.md).
The [calendar replay](bard-blrm-trials.md) combines these decisions with
posterior fitting and explicitly supplied outcomes and delays. [Stage-two
continuation](bard-two-stage.md) integrates eligible carryover and complete
outcomes. [Accelerated titration](bard-titration.md) supplies the guide's prefix
under explicit scheduling and BF-BLRM safety conventions. Expansion and
stage-two calendar timing remain open.
