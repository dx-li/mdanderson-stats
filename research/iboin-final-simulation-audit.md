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
