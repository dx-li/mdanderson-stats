# Full waterfall workflow implementation audit

This checkpoint follows the source investigation in
[boin-waterfall-next-audit.md](boin-waterfall-next-audit.md). The comparison and
desktop mapping are integrated as `6676ba3` and `e96fbb6`; the next implementation
is assigned to the same Luna worker in its separate existing checkout. There
is one implementation worker and numerical jobs run serially with one thread.

## Independent reference design

The six original-R subtrial fixtures cover local conduct, including precision
stopping after movement to a previously treated destination. The additional
`tools/reference_boin_waterfall_workflow.R` loads the same pinned BOIN 2.7.2
source and exercises its full wrapper with specified cohort outcomes. Calls
record each subtrial's space, start, budget, candidate, enrollment, DLTs and
exclusions, plus the actual assigned-dose history.

The temporary R `Iso` installation used in earlier audits is absent. Instead
of adding another installation, the new generator replaces only its bivariate
fit call with an independent six-cell oracle. It enumerates all subsets of
the seven neighboring order constraints on a 2-by-3 grid, pools the connected
components of active equality constraints using weighted means, and selects
the feasible fit with minimum weighted squared loss. Every isotonic optimum
has such an active equality set. Original conduct and contour-selection
statements remain unchanged. These are original-R workflow references with an
independent exhaustive fit, not outputs from an installed `Iso` package.
Previously generated `Iso::biviso` fixtures separately validate the shared
Python bivariate primitive.

Planned cases cover ordinary two-row traversal, failed first and later
subtrials, the special same-row branch, its fallback after a failed subtrial,
and an earlier-row candidate that does not permit escalation. The native
outer loop can discard the just-enrolled patients of an unsuccessful ordinary
subtrial. Actual dose histories and returned native counts must therefore be
recorded separately. A Python implementation must retain the actual patients.

## Selection policy requiring an explicit contract

Native final selection pools all grid cells with weights `n+.1`, substitutes
1.1 at eliminated cells before fitting, and adds a `1e-5` row-plus-column tie
adjustment. Initial row candidates come from treated, noneliminated cells.
Continuity can move a row's candidate to the column chosen in the following
row. Untreated-row and inadmissible continuity destinations must not silently
acquire a different statistical rule: the Python result should expose both
the candidates and any explicit reason for withholding a final recommendation.

The original article is Zhang and Yuan (2016), DOI
[10.1002/sim.7095](https://doi.org/10.1002/sim.7095). A current search confirmed
the paper identity; the PMC full-text page returned a browser challenge, so
this audit does not infer a paper rule from inaccessible text. The pinned R
source supplies the current implementation contract.

## Reference results

The workflow generator completed in 0.50 seconds. Its six cases are recorded in
`tests/fixtures/boin-waterfall-workflows.csv`, with `-workflow-traces`,
`-workflow-subtrials` and `-workflow-fits` companion files. The existing Python
bivariate fitter agrees with all six exhaustive fits within `1.74e-17` absolute
error. The corrected one-off check used warnings-as-errors and closed its CSV
input; an initial check emitted a file-resource warning and was rerun after
correcting the verification script's file handling.

The first failed-subtrial case actually enrolled 3 patients, all with DLTs;
the native wrapper returned zero patients and zero DLTs. In the later-failure
case, the actual trace records 18 patients and 6 DLTs, but native output keeps
only the preceding 12 patients and zero DLTs. It additionally recommends an
unobserved upper-row dose through its continuity rule. These exact differences
are kept in the reference files, and the Python port must not reproduce the
loss of observations. Four other cases return complete counts and provide
direct final-selection comparisons.

## Integrated Python workflow

Luna implemented the workflow in `a3a07bd`, integrated as `3b5ab30`.
`run_boin_waterfall_trial` consumes an explicit per-patient uniform tape;
`simulate_boin_waterfall` generates tapes serially. Both cover the staircase,
ordinary row searches, special same-row search and its failed-subtrial fallback.
Results retain actual observations, cumulative subtrial snapshots, cohort
assignments, exclusions, candidates and source/final contour diagnostics.
The workflow's precision threshold is checked after movement, independently
of the ordinary combination design's stopping parameter.

Python preserves every enrolled patient, uses actual simulation denominators,
honors supplied boundary parameters and withholds inadmissible continuity
destinations while retaining their source candidates. It does not claim full
native-output parity. Titration, the app's 3+3 run-in and generated reports
remain outside this API. Catalog entries 99 and 128 remain partial.

## Validation and resource use

The worker passed three focused waterfall checks and two comparison checks,
Ruff, formatting and targeted mypy. Root independently checked all six original
subtrial cases and all six workflow traces, including exact actual patient/DLT
accounting in the two defective native cases. The four cases without source
count loss match final selections, exclusions and unrounded fits. An additional
3-by-3 traversal and independent aggregation of 32 simulated trials checked
row selection/no-selection probabilities, Monte Carlo errors and count totals.
An oversized request fails before advancing the supplied random generator.

The bounded root check completed in 0.043 seconds after import, with peak
resident memory 112.30 MiB and zero swaps. An initial verification adapter
mistakenly treated the final API's full-grid subtrial snapshots as local vectors;
correcting that adapter resolved its shape error without an algorithm change.

The wheel and source distribution were built with cached Hatch tooling. An
isolated wheel import verified all eleven new comparison/waterfall exports and
executed four documentation examples. That check took 0.108 seconds after import,
peaked at 115.02 MiB and recorded zero swaps. Packaged module/catalog bytes and
third-party notices match the source. Focused formatting/lint checks passed.
The full repository suite and large Monte Carlo runs were intentionally omitted;
checks targeted source numerical behavior and the new public workflows. Jobs
were serial, with one BLAS/OpenMP thread, and no dependency installation.

Source review identified another conditional difference: native interim
extra-safe evaluation sits inside `!is.na(b.elim[n])`. At high targets/cutoffs,
no ordinary elimination count may exist at that sample size, although the
weaker extra-safe condition could stop. Python deliberately applies the
specified extra-safe rule whenever `n>=3`, without that source availability
gate, and retains final-subtrial safety exclusions. These are documented
safety contracts, not claims of bit-for-bit native behavior for all settings.
