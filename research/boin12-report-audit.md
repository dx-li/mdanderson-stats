# BOIN12 report workflow audit

The cached BOIN12 guide advertises operating-characteristic simulation and
report generation; the existing Python `simulate_boin12` and
`simulate_boin12_two_stage` functions already provide the underlying validated
conduct engines. This addition supplies a reproducible saved workflow around
those APIs without changing their statistical rules.

`boin12_report` captures a reconstructed `BOIN12Design`, full four-cell joint
truth tables for uniquely labeled scenarios, simulation settings, optional
Stage 1 threshold, and an integer NumPy seed. It runs scenarios serially and
reduces each result to OBD/MTD probabilities and Bernoulli MCSEs (including
the no-selection index), per-dose average counts, stop-reason frequencies,
and—when two-stage mode is requested—transition and stage-cohort summaries.
The report retains no trial-level simulation matrices. Aggregate scenario,
trial, planned-patient, dose-cell, and work limits are checked before the first
simulation call.

The native app's optional 3+3 run-in precedence when a 1/3 pattern interacts
with utility selection is not resolved by recovered source. This report
faithfully records the existing Python conduct policy and labels that
limitation rather than silently selecting a native rule. Native formatting,
exports, and RNG sequence are outside this report's claim. It implements no
new statistical calculation or multilevel endpoint method.
