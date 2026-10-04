# RF-SRC missing-response OOB reference

This independent fixed-tape reference narrows the pinned RF-SRC 3.2.2
missing-data audit to response completion for OOB performance and concordance.
It does not execute or redistribute the native forest and does not claim native
RNG parity.

## Provenance

The source pin is CRAN mirror revision
`b4d099e262423362a8872c13c468e6dbe2f9e9da`; see
`research/random-survival-forest-audit.md` and
`research/random-survival-missing-reference-audit.md` for version, license,
and cached-file blob hashes. The reference uses equations from the cached
`src/randomForestSRC.c`, not copied source code.

## Contract represented

`stackAndImputePerfResponse` (`src/randomForestSRC.c:11265–11318`) creates a
temporary response copy for grow-time performance. It calls
`imputeResponse` (`5682–5695`), which calls `imputeCommon` with OOB selection
(`selectionFlag=FALSE`; `5698–5930`). Only originally missing response
components are changed. For a missing row and response field, the primary
donor values are terminal completions from trees where that row was OOB. Each
tree's terminal time completion has already been snapped to the master time
grid by terminal imputation (`4991–4995`, stored in terminal membership at
`5052–5057`); a nonempty OOB pool averages these snapped values without
snapping the ensemble mean a second time. A nonempty status pool is reduced to
its mode with random choice among tied modes. If the primary pool is
empty, the source samples a value from original full-data observations where
that field is present; an empty response fallback pool is fatal. Tree status
ties and global fallback draws use the native helper's uniform-tape convention
in the R reference, not R or NumPy seeded streams. All nonempty tree-pool
aggregations (including tied status choices) precede the fallback pass; time
fallback sampling precedes status fallback sampling. Final response summaries use
a separate time-grid snap path and are not the response passed to OOB
concordance.

`getPerformance` (`11328ff`) supplies the completed response, OOB mortality,
and contributor counts to `getConcordanceIndex` (`32500–32548`). A pair is
ignored unless both rows have OOB contributors. Comparable pairs include an
earlier failure versus a later observation, failure/censor time ties within
`EPSILON`, and tied failure/failure times. The source's tie and inversion
increments are preserved explicitly in the pair ledger; no-comparable-pair
output is NaN.

## Scope

`tools/reference_random_survival_missing_oob.R` writes small CSV ledgers for
means of snapped per-tree OOB times left unsnapped at ensemble level, status
mode ties, original-observation fallback, preservation of observed components,
and concordance pair classification. The time-mean case uses master-grid
values 1 and 3, whose ensemble mean 2 is absent from the grid, plus an in-bag
terminal-time outlier of 100 that must not enter the recipient's OOB pool.
They are hand-specified source-equation cases, not full-forest simulations.
The Python checker compares the engine's deterministic private aggregation
helper and native-convention concordance calculation against these immutable
values. A full fit may use a distinct NumPy stream, and aggregate tie outcomes
are consequently not expected to match native seeded runs.

## Integrated validation, 2026-10-04

All 75 affected forest, OOB, Brier, importance and reference checks pass with
warnings as errors in 3.730 seconds, with 144.31 MiB peak process RSS and zero
swaps. The R ledgers regenerated and all five direct OOB reference comparisons
pass. Separate public-fit checks cover global fallback with no OOB trees and
representable means of terminal times near `1e308`. Predictor-imputed Brier
contributions match independent global censor/IPCW equations, with complete
retained outcomes, dropped all-missing rows and stale-row-map rejection.

Scoped Ruff, formatting and mypy checks pass for the integrated source. The
engine retains per-leaf response scalars and row routes, without a dense
tree-by-row response cube. OOB time means accumulate stably, status-mode ties
precede field-ordered fallback draws, and resource preflights include the added
state. Numerical and static checks remain serial with single-threaded numerical
libraries; no full local suite, installation or new CI workflow was added.
