# BOIN waterfall titration source audit

The first-subtrial titration prelude is implemented from the pinned BOIN 2.7.2
source `research/raw/BOINComb/R/get.oc.comb.R` (`waterfall.subtrial` and
`get.oc.comb.waterfall`; source MD5 `342645d339cc8bcb47568f2bd76a1069`). The
existing original-R wrapper harness is
[`tools/reference_boin_waterfall_titration.R`](../tools/reference_boin_waterfall_titration.R);
its generated fixtures are under `tests/fixtures/boin-waterfall-titration*`.

In the first waterfall subtrial, the source constructs a staircase down the
first dose column and then across the last row. When titration is enabled and
cohort size exceeds one, it draws one potential outcome for every staircase
position at once. It assigns one patient to each visited position through the
first DLT; if no DLT occurs, it assigns one to the whole staircase and ends at
the highest position. Outcomes drawn after a first DLT are unused for patient
counts, but are still consumed by that source vector draw. On the first
ordinary iteration at the endpoint, the source adds `cohort_size - 1` patients
once, then applies its usual elimination, movement, early-stop, and cohort
budget rules. Cohort size one disables titration.

The wrapper starts the first staircase at its first position; it does not use
the `startdose` argument for a titration prelude. The Python replay retains this
source behavior and reports the endpoint separately from the first
subtrial's `start_position`. The replay tape reserves a prefix as long as the
full staircase so unused potential outcomes cannot shift subsequent cohort
outcomes. A simulation generates this fixed-size tape one trial at a time.
With titration off, the previous tape length and draw sequence are unchanged.

The source subtrial returns both an `ntotal` field computed as
`icohort * cohort_size` and cumulative matrices. The first quantity does not
count actual titration enrollment correctly. The outer wrapper instead uses
`sum(npts)` for its global planned-enrollment check. Python reports and checks
actual retained counts and preserves all assigned patients. Since the wrapper
checks its global total after a started subtrial, enrollment may exceed the
planned number of patients by at most `staircase_length - 1`; this overhead is
included in replay-tape, simulation-work and storage preflight. Python caps
planned enrollment plus that overhead at 1,000 possible actual patients.

Validation should compare the per-patient assignments, not only native summary
matrices: another native failure/safety path can discard actual observations
from returned counts. The fixed-outcome R fixtures retain both the native
returned counts and the complete assigned-patient trace. Python intentionally
preserves the latter as actual counts. No native RNG algorithm parity is
claimed; uniform-tape assignments and titration-off seed behavior are the
reproducibility contract.

The focused check `pytest -q tests/test_boin_waterfall_trial.py -W error`
passed all 8 tests. It compares all 7 fixed original-R cases across 43 assigned
records and 11 subtrial snapshots, including dose spaces, starting positions,
actual counts, exclusions, candidates and escalation flags. The final native
exclusion mask is compared except for the first-DLT safety case where the
native wrapper returns a mask after dropping its actual three-patient
subtrial; Python retains the actual DLT counts and safety state. Separate
checks cover deterministic titration endings, cohort-size-one disabling,
seed replay, count conservation and rejection before advancing the simulation
generator when the extra-staircase bound is exceeded. The run completed in
2.114 seconds with peak `ru_maxrss` 134,725,632 bytes and zero swaps. Targeted
Ruff lint/format and mypy checks passed. No full suite or large simulation was
run.
