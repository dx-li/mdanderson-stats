# iBOIN final selection and simulation source audit

The recovered final-selection help documents the unborrowed final rate as
`y/n` and the borrowed rate as `(y + m*q)/(n + m)`, with `q` the skeleton and
`m` the prior effective sample size used for the estimate. It does not resolve
the isotonic regression weights, tie rules, how untreated doses participate,
the use of robust effective rather than original prior ESS, or all behavior of
the optional final upper-bound setting. The implementation therefore exposes
those as caller-selected policies and does not claim native application parity.

The selection calculation fits the weighted increasing isotonic regression
across treated doses, including treated doses later excluded from candidacy.
Only treated, noneliminated doses passing the optional eligibility mask and
optional per-dose final de-escalation bound may be selected. The lowest dose
wins distances tied within `1e-14` by default. Safety elimination and the
design's extra-safe terminal rule always suppress selection.

The simulator reuses the package's deterministic iBOIN replay transition for
titration, complete-cohort decisions, elimination, and terminal stops. Its
severity generation assumes the explicit caller-supplied grade-2 and DLT
probabilities are mutually exclusive maximum-severity categories. The
aggregate runner uses serial per-trial seeds, no retained patient histories,
and bounded preflight work/storage. Native RNG, event-association assumptions,
selection conventions, and native operating-characteristic report parity are
not established by the recovered sources.

Focused tests cover raw and borrowed rate equations, weighted isotonic fitting
and candidate filtering, tie policy, safety suppression, one-trial conduct,
seed replay, category and count conservation, resource-policy validation, and
undefined Monte Carlo errors for a single repetition. Independent exact
isotonic-reference and replay-differential checks are coordinated separately
by the project integrator.

## Independent integration checks

The standard-library reference generator `tools/reference_iboin_isotonic.py`
enumerates contiguous partitions and minimizes exact rational weighted squared
error without NumPy, SciPy or package imports. All 36 scenarios (114 dose rows)
in `tests/fixtures/iboin-isotonic-exact.csv` match the implemented projection;
maximum absolute floating-point difference is 5.56e-17. These checks cover
observed/borrowed rates and patient/effective/custom weights, not native defaults.

A bounded comparison against published commit `61f151f` checks all 1,456
severity sequences of lengths zero through five across ordinary cohorts,
titration, lower-cap handoff and single-patient cohorts. Every replay field,
readonly array and post-stop rejection matches. Twelve separately replayed
trial seeds reproduce aggregate selections, stops, mean counts and count MCSEs;
robust-effective ESS and empty final-bound eligibility also pass. The combined
independent run took 1.094 seconds, peaked at 124.80 MiB and reported zero swaps.

The integrated selector/simulation suite passes nine focused checks after the
final explicit-array typing adjustments, with warnings treated as errors:
1.868 seconds, 146.09 MiB peak RSS and zero swaps. Targeted type, lint and
format checks pass for both new workflows.
