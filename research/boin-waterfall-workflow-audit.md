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

The final Python API and its verification will be recorded after integration.
