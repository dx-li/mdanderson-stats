# RF-SRC iterated survival imputation reference

This independent fixed-tape reference records scalar imputation summaries and
pass transitions for `randomForestSRC` 3.2.2. It is not a native forest
execution and does not claim parity between the native random streams and
NumPy.

## Provenance

The source pin is CRAN mirror revision
`b4d099e262423362a8872c13c468e6dbe2f9e9da`; see
`research/random-survival-forest-audit.md` for source hashes and the GPL
provenance note. The reference is derived from the ignored cached source and
does not redistribute native source.

## Contract exercised

In an ordinary fit, native `grow` enters one pass loop (`src/randomForestSRC.c`
`10600ff`). Pass 1 grows using original missingness and terminal donor fills;
after pass 1 it computes a missing-cell summary even when `nimpute=1`
(`10687–10731`). If another pass follows, `acquireTreeGeneric` applies those
values only at the original missing cells before growing (`33939ff`,
`imputeUpdateShadow` `5576–5667`). Intermediate passes refit on the updated
data. The final pass's trees and curves are retained; ordinary fitting does
not apply a post-final summary. Thus with `nimpute=3`, pass-2 summaries provide
the data for pass 3, while pass-3 terminal summaries do not replace the
training values or fitted final curves. `impute.only` has a separate final
summary path and is outside this reference.

`imputeCommon` pools one scalar terminal summary per eligible tree and missing
cell (`5812–5863`), preserving OOB vs all-tree eligibility. By-root/by-user
selection uses OOB terminal values; by-node/no-bootstrap uses the all-tree
selection. Numeric/time fields use means, categorical/status fields use modes;
empty pools fall back to original observed values for the field, with outcome
errors if no fallback donor exists. Time terminal values are snapped per tree;
after pooling, missing times are snapped again to the fixed master grid by
`imputeMultipleTime` (`5933–5962`). The master grid comes from every original
finite time, including censored rows and rows with missing status; it is not
rebuilt between passes. The event-interest grid remains based on complete
observed events. Original missing masks remain authoritative across passes,
and observed values are not overwritten.

The R ledger supplies explicit uniforms only for donor selection and ties. It
does not attempt native chain initialization, bootstrap generation, or exact
seeded replay. The Python engine's repeatable NumPy seed contract is separate.

## Fixture scope

`tools/reference_random_survival_iterated.R` produces scalar summary and pass
ledgers, a noncontiguous row map, and a coupled OOB-pooling tape. The coupled
case includes row-major status ties, variable-major global fallback, and a
second time snap whose output differs from the pooled mean. An imputed event
at a time outside the original complete-event grid makes rebuilding that grid
detectably wrong. These are direct semantic references; the pass ledger only
records which state each stage consumes. It does not verify the forest's
orchestration or model-fitting outcomes, and none of the artifacts imply
whole-forest native RNG or numerical parity.

The focused Python comparison also calls the public forest fitter for two
imputation passes with a fully missing row removed from the analyzed data.
With fixed NumPy seed 1, an originally missing event at time 4 is imputed as
an event, while the retained event-interest grid remains the original
`[1, 2, 5, 7]`. This checks the row map, completed arrays, pass metadata and
the distinction between the fixed original grid and the completed outcomes;
it is a Python engine integration check, not independent native numerical
parity.
