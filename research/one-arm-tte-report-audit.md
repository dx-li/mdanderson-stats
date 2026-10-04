# OneArmTTE scenario report source audit

Audit date: 2026-10-04. Sources are the cached version 3.0.9 embedded help
files named in `docs/one-arm-tte-sources.json`; no vendor endpoint was accessed
for this work.

## Workflow and model crosswalk

The statistical computation was already present before this report workflow.
`src/mdanderson_stats/one_arm_tte.py` defines the inverse-gamma-updated
single-arm monitor and deterministic calendar conduct; the corresponding model
and strict rule semantics are described in `docs/one-arm-tte-source.md`.
`src/mdanderson_stats/one_arm_tte_simulation.py` simulates exponential event
durations and Poisson arrivals, and returns rule frequencies and binomial
MCSEs, sample-size and final-time quantiles, and per-replication patient,
event, exposure, accrual-stop/final-time and rule-flag arrays.

Cached help `_C6097499C47C8703F7FB028BB33EE844.txt` lines 208–245 defines
scenario name, true mean/median TTE, accrual rate and positive RNG seed; it
states that seeds reset per scenario and scenario order is retained in output.
`_36DEB89ED76E8990197287DE35B35C31.txt` lines 1–21 defines repetitions,
credible-interval percentage and at least one scenario. `_F00E562F8E2C371B583A10679BD47280.txt`
lines 1–20 describes the report's design-input section, one summary row per
scenario, and ordered scenario-specific result sections. It does not specify
the exact report table columns. The same scenario guide recommends evaluating
mean patient count and trial duration (see `_C6097499C47C8703F7FB028BB33EE844.txt`
lines 94–98 and 184–195).

The Python report therefore captures the actual `OneArmTTEDesign` settings,
validates all scenario inputs and aggregate potential-patient work before
simulation, runs scenarios serially with their own seeds, and summarizes the
existing per-replication outputs. It derives means from the existing returned
vectors and retains compact immutable summaries rather than all trial arrays.
The simulator now exposes its actual accrual-phase check count and accepts a
bounded remaining-check budget, allowing a report to enforce a cumulative
monitoring ceiling without assuming stochastic periodic-check counts in
advance. The final assessment is separate and excluded from this count, as in
the simulator's existing result contract.

## Deliberate boundaries

No missing statistical method was identified. The user-facing gap was a
repeatable multi-scenario run and readable saved output. Python HTML is a
standalone view, not a parser or editable native project/session. It does not
claim native report columns beyond those supported by the inspected help,
Windows formatting, or matching native random streams. The strict cutoffs,
overlapping rule flags, NumPy stream, and quantile interpolation remain those
of the existing Python simulation API.
